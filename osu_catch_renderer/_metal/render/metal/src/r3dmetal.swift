import Metal
import Foundation

// R3D Metal backend — C ABI for ctypes from Python.
// The framebuffer is a buffer-backed MTLTexture over .storageModeShared, so the
// CPU reads rendered pixels through a pointer with NO copy. Measured 0.116 ms
// for 1920x1080 under 92% CPU load, vs 1.38-1.56 ms for the GL PBO path.

// Ring depth is a CREATE-TIME parameter, not a constant. A frame returned by
// r3dm_acquire is a view into its ring slot and stays valid only until that slot
// is reused, so the ring must cover every downstream reference: composite
// queue(3) + in-process(1) + writer queue(12) + in-write(1) + slack. Same budget
// gl.py's _HOST_POOL=24 exists for; undersizing it silently corrupts output
// (measured on the GL side: ring=12 changed the mp4 md5).
private let RING_DEFAULT = 3

public final class Ctx {
    let dev: MTLDevice
    let queue: MTLCommandQueue
    let w: Int, h: Int
    let bpr: Int, bufLen: Int
    let ring: Int
    var bufs: [MTLBuffer] = []
    var texes: [MTLTexture] = []
    var inflight: [MTLCommandBuffer?]
    var head = 0                      // frames submitted
    var tail = 0                      // frames handed to the CPU
    var cur: MTLCommandBuffer?        // command buffer being built
    var enc: MTLRenderCommandEncoder? // stays open across draws in a frame
    var lastErr: String = ""
    var death: DeathKit?
    var hp: HpKit?
    var hpi: HpInline?
    var pill: PillKit?
    var yuv: YuvKit?
    // One yuv420p output per RING SLOT, so render+convert can be chained into a
    // single command buffer and collected later -- the CPU never waits.
    var yuvRing: [MTLBuffer] = []
    // rgba8Uint VIEWS of the render targets. Reading the backing
    // buffer instead let the compute kernel race the render pass --
    // Metal does not track that a texture and its buffer alias, but
    // it DOES track a texture view of that texture.
    var texesUint: [MTLTexture] = []
    var flash: FlashKit?
    var pipeNormal: MTLRenderPipelineState!
    var pipeAdd: MTLRenderPipelineState!
    var smp: MTLSamplerState!
    var texes2: [Int32: MTLTexture] = [:]     // sprite textures by Python-side id
    var nextTex: Int32 = 1
    // Per-ring instance buffers. 14 floats/instance, matching gl.py _I_STRIDE.
    // Ringed so a draw never writes a buffer the GPU is still reading.
    var instBufs: [MTLBuffer] = []
    var instOff = 0                   // byte offset within this frame's buffer
    static let INST_STRIDE = 14 * 4
    static let INST_CAP = 16384

    init?(w: Int, h: Int, ring: Int) {
        guard let d = MTLCreateSystemDefaultDevice(),
              let q = d.makeCommandQueue() else { return nil }
        dev = d; queue = q; self.w = w; self.h = h
        self.ring = max(3, ring)
        let align = d.minimumLinearTextureAlignment(for: .rgba8Unorm)
        bpr = ((w * 4) + align - 1) / align * align
        bufLen = bpr * h
        inflight = Array(repeating: nil, count: self.ring)
        let td = MTLTextureDescriptor()
        td.pixelFormat = .rgba8Unorm
        td.width = w; td.height = h
        // NOT .pixelFormatView. Adding it to allow an rgba8Uint view for the
        // on-die yuv path disabled lossless framebuffer compression and HALVED
        // gameplay throughput on the 3408-object map (579.5 -> 280.9 fps) --
        // for a feature that is off by default. If the on-die path is ever
        // revived, gate this flag on it rather than paying it always.
        td.usage = [.renderTarget, .shaderRead]
        td.storageMode = .shared
        for _ in 0..<self.ring {
            guard let b = d.makeBuffer(length: bufLen, options: .storageModeShared),
                  let t = b.makeTexture(descriptor: td, offset: 0, bytesPerRow: bpr)
            else { return nil }
            bufs.append(b); texes.append(t)
            guard let ib = d.makeBuffer(length: Ctx.INST_CAP * Ctx.INST_STRIDE,
                                        options: .storageModeShared)
            else { return nil }
            instBufs.append(ib)
        }
        guard let (pn, pa, sm) = r3dmMakePipelines(d, .rgba8Unorm) else { return nil }
        pipeNormal = pn; pipeAdd = pa; smp = sm
    }
}

private var ctxs: [Int32: Ctx] = [:]

/// Accessor so the death kit (a separate file) can reach a context.
func r3dmCtx(_ id: Int32) -> Ctx? { return ctxs[id] }
private var nextId: Int32 = 1
private let lock = NSLock()

@_cdecl("r3dm_create")
public func r3dm_create(_ w: Int32, _ h: Int32, _ ring: Int32) -> Int32 {
    let rr = ring > 0 ? Int(ring) : RING_DEFAULT
    guard let c = Ctx(w: Int(w), h: Int(h), ring: rr) else { return -1 }
    lock.lock(); defer { lock.unlock() }
    let id = nextId; nextId += 1
    ctxs[id] = c
    return id
}

@_cdecl("r3dm_info")
public func r3dm_info(_ id: Int32, _ outBpr: UnsafeMutablePointer<Int32>,
                      _ outLen: UnsafeMutablePointer<Int32>) -> Int32 {
    guard let c = ctxs[id] else { return -1 }
    outBpr.pointee = Int32(c.bpr); outLen.pointee = Int32(c.bufLen)
    return 0
}

@_cdecl("r3dm_device_name")
public func r3dm_device_name(_ id: Int32, _ out: UnsafeMutablePointer<CChar>,
                             _ cap: Int32) -> Int32 {
    guard let c = ctxs[id] else { return -1 }
    let s = "\(c.dev.name)|unified=\(c.dev.hasUnifiedMemory)"
    _ = s.withCString { strlcpy(out, $0, Int(cap)) }
    return 0
}

/// Begin a frame: open a command buffer and clear the current ring target.
@_cdecl("r3dm_begin")
public func r3dm_begin(_ id: Int32, _ r: Float, _ g: Float, _ b: Float,
                       _ a: Float) -> Int32 {
    guard let c = ctxs[id] else { return -1 }
    guard let cb = c.queue.makeCommandBuffer() else { return -2 }
    let k = c.head % c.ring
    let rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = c.texes[k]
    rp.colorAttachments[0].loadAction = .clear
    rp.colorAttachments[0].storeAction = .store
    rp.colorAttachments[0].clearColor = MTLClearColor(
        red: Double(r), green: Double(g), blue: Double(b), alpha: Double(a))
    guard let e = cb.makeRenderCommandEncoder(descriptor: rp) else { return -3 }
    var screen = SIMD2<Float>(Float(c.w), Float(c.h))
    e.setVertexBytes(&screen, length: 8, index: 1)
    c.cur = cb; c.enc = e; c.instOff = 0
    return 0
}

/// Upload a sprite texture (RGBA8, tightly packed). Returns its id, or -1.
@_cdecl("r3dm_tex_create")
public func r3dm_tex_create(_ id: Int32, _ w: Int32, _ h: Int32,
                            _ px: UnsafeRawPointer, _ mip: Int32) -> Int32 {
    guard let c = ctxs[id] else { return -1 }
    let td = MTLTextureDescriptor()
    td.pixelFormat = .rgba8Unorm
    td.width = Int(w); td.height = Int(h)
    td.usage = [.shaderRead]
    td.storageMode = .shared
    // MIPMAPS ARE NOT OPTIONAL. The GL path uses mipmapped LINEAR, and the
    // background is a full-screen heavily-downscaled image; plain bilinear
    // aliases against it badly (measured: PSNR 19-28 dB vs GL without mips).
    let levels = Int(floor(log2(Double(max(w, h))))) + 1
    td.mipmapLevelCount = mip != 0 ? levels : 1
    guard let t = c.dev.makeTexture(descriptor: td) else { return -1 }
    t.replace(region: MTLRegionMake2D(0, 0, Int(w), Int(h)),
              mipmapLevel: 0, withBytes: px, bytesPerRow: Int(w) * 4)
    if mip != 0 && levels > 1 {
        guard let cb = c.queue.makeCommandBuffer(),
              let bl = cb.makeBlitCommandEncoder() else { return -1 }
        bl.generateMipmaps(for: t)
        bl.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()      // upload-time only, never per frame
    }
    let tid = c.nextTex; c.nextTex += 1
    c.texes2[tid] = t
    return tid
}

/// One instanced draw. `inst` is count x 14 floats (centre, size, rot, colour,
/// uv_off, uv_scale, tex_unit) -- byte-for-byte the GL path's layout. `texIds`
/// maps unit 0..nTex-1 to uploaded texture ids. `additive != 0` selects the
/// additive blend pipeline. Painter's order within the call is preserved.
@_cdecl("r3dm_draw")
public func r3dm_draw(_ id: Int32, _ inst: UnsafeRawPointer, _ count: Int32,
                      _ texIds: UnsafePointer<Int32>, _ nTex: Int32,
                      _ additive: Int32) -> Int32 {
    guard let c = ctxs[id], let e = c.enc else { return -1 }
    let n = Int(count)
    if n <= 0 { return 0 }
    let bytes = n * Ctx.INST_STRIDE
    let k = c.head % c.ring
    if c.instOff + bytes > Ctx.INST_CAP * Ctx.INST_STRIDE { return -2 }
    memcpy(c.instBufs[k].contents().advanced(by: c.instOff), inst, bytes)
    e.setRenderPipelineState(additive != 0 ? c.pipeAdd : c.pipeNormal)
    e.setVertexBuffer(c.instBufs[k], offset: c.instOff, index: 0)
    // Re-bind the viewport EVERY draw. r3dm_begin binds it once, but the
    // in-pass HUD shaders (hp_inline, pill_inline) bind their own params over
    // these slots, and a sprite draw after one of them read a stale/nil
    // `screen` and collapsed every quad to zero size -- silently, since the
    // in-pass HUD used to be the last thing in the pass. 8 bytes per call.
    var screen = SIMD2<Float>(Float(c.w), Float(c.h))
    e.setVertexBytes(&screen, length: 8, index: 1)
    e.setFragmentSamplerState(c.smp, index: 0)
    for u in 0..<Int(nTex) {
        e.setFragmentTexture(c.texes2[texIds[u]], index: u)
    }
    e.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4,
                     instanceCount: n)
    c.instOff += bytes
    return 0
}

/// Commit the frame. Returns 1 if a completed frame is ready (ring filled), else 0.
@_cdecl("r3dm_commit")
public func r3dm_commit(_ id: Int32) -> Int32 {
    guard let c = ctxs[id], let cb = c.cur else { return -1 }
    c.enc?.endEncoding(); c.enc = nil
    cb.commit()
    c.inflight[c.head % c.ring] = cb
    c.head += 1
    c.cur = nil
    return (c.head - c.tail >= c.ring) ? 1 : 0
}

/// Wait for the OLDEST in-flight frame and return a pointer to its pixels.
/// Zero-copy: this is the shared buffer the GPU rendered into.
///
/// `force == 0` returns nil until the ring is FULL, so the GPU always has
/// RING-1 frames of slack. That pipelining is not optional: waiting on the
/// frame just submitted measured 0.302 ms vs 0.122 ms pipelined. Mirrors the
/// GL path's read_rgb_async(); `force != 0` is read_drain().
@_cdecl("r3dm_acquire")
public func r3dm_acquire(_ id: Int32, _ force: Int32) -> UnsafeMutableRawPointer? {
    guard let c = ctxs[id], c.tail < c.head else { return nil }
    if force == 0 && (c.head - c.tail) < c.ring { return nil }
    let k = c.tail % c.ring
    c.inflight[k]?.waitUntilCompleted()
    c.inflight[k] = nil
    c.tail += 1
    return c.bufs[k].contents()
}

@_cdecl("r3dm_destroy")
public func r3dm_destroy(_ id: Int32) -> Int32 {
    lock.lock(); defer { lock.unlock() }
    guard let c = ctxs[id] else { return -1 }
    for cb in c.inflight { cb?.waitUntilCompleted() }
    ctxs.removeValue(forKey: id)
    return 0
}
