import Metal

// GPU apply_death: colour grade then affine warp, as TWO passes.
// One pass would be wrong -- PIL grades first (clamping at 255) and then
// interpolates, and clamp is non-linear, so bilinear-of-graded != graded-of-
// bilinear wherever the additive red pushes a channel over 255.
let R3DM_DEATH_SRC = """
#include <metal_stdlib>
using namespace metal;

struct VO { float4 pos [[position]]; float2 uv; };
constant float2 Q[4] = { float2(0,0), float2(1,0), float2(0,1), float2(1,1) };

vertex VO death_vs(uint vid [[vertex_id]]) {
    float2 q = Q[vid];
    VO o;
    o.pos = float4(q.x*2.0-1.0, 1.0-q.y*2.0, 0.0, 1.0);
    o.uv  = q;
    return o;
}

struct Grade { float mul; float redAdd; };

// pass 1: FadeColour(Gray) darken + additive red wash, clamped -- matches
// death.py's LUT exactly (v*mul, +redAdd on R, clamp 0..255).
fragment float4 death_grade_fs(VO in [[stage_in]],
                               texture2d<float> src [[texture(0)]],
                               sampler smp [[sampler(0)]],
                               constant Grade& g [[buffer(0)]]) {
    float4 c = src.sample(smp, in.uv);
    float3 v = c.rgb * 255.0 * g.mul;
    v.r += g.redAdd;
    // TRUNCATE, don't round. numpy's .astype(uint8) truncates; writing a
    // float to a unorm8 target rounds to nearest, which showed up as a uniform
    // ~0.5-level bias (mean |diff| 0.74, PSNR capped at 48 dB). floor() first,
    // then the /255 value is an exact integer and the unorm write is lossless.
    return float4(floor(clamp(v, 0.0, 255.0)) / 255.0, 1.0);
}

struct Warp { float a, b, c, d, e, f; float w, h; };

// pass 2: PIL's AFFINE maps OUTPUT->INPUT, same coefficients death.py computes.
fragment float4 death_warp_fs(VO in [[stage_in]],
                              texture2d<float> src [[texture(0)]],
                              sampler smp [[sampler(0)]],
                              constant Warp& t [[buffer(0)]]) {
    float2 o = float2(in.uv.x * t.w, in.uv.y * t.h);
    float2 i = float2(t.a*o.x + t.b*o.y + t.c, t.d*o.x + t.e*o.y + t.f);
    if (i.x < 0.0 || i.y < 0.0 || i.x >= t.w || i.y >= t.h)
        return float4(0.0, 0.0, 0.0, 1.0);          // black fill, as PIL
    return float4(src.sample(smp, float2(i.x / t.w, i.y / t.h)).rgb, 1.0);
}
"""

final class DeathKit {
    let grade: MTLRenderPipelineState
    let warp: MTLRenderPipelineState
    let smp: MTLSamplerState
    let srcBuf: MTLBuffer, midBuf: MTLBuffer, dstBuf: MTLBuffer
    let srcTex: MTLTexture, midTex: MTLTexture, dstTex: MTLTexture
    let w: Int, h: Int, bpr: Int, len: Int

    init?(dev: MTLDevice, w: Int, h: Int) {
        self.w = w; self.h = h
        let align = dev.minimumLinearTextureAlignment(for: .rgba8Unorm)
        let lbpr = ((w * 4) + align - 1) / align * align
        let llen = lbpr * h
        bpr = lbpr; len = llen
        guard let lib = try? dev.makeLibrary(source: R3DM_DEATH_SRC, options: nil),
              let vs = lib.makeFunction(name: "death_vs"),
              let fg = lib.makeFunction(name: "death_grade_fs"),
              let fw = lib.makeFunction(name: "death_warp_fs") else { return nil }
        func pipe(_ fs: MTLFunction) -> MTLRenderPipelineState? {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = vs; d.fragmentFunction = fs
            d.colorAttachments[0].pixelFormat = .rgba8Unorm
            return try? dev.makeRenderPipelineState(descriptor: d)
        }
        guard let pg = pipe(fg), let pw = pipe(fw) else { return nil }
        grade = pg; warp = pw
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear; sd.magFilter = .linear
        sd.sAddressMode = .clampToEdge; sd.tAddressMode = .clampToEdge
        guard let sm = dev.makeSamplerState(descriptor: sd) else { return nil }
        smp = sm
        let td = MTLTextureDescriptor()
        td.pixelFormat = .rgba8Unorm; td.width = w; td.height = h
        td.usage = [.renderTarget, .shaderRead]; td.storageMode = .shared
        // NOTE: locals only -- a closure touching self.len/self.bpr here would be
        // "self used before all stored properties are initialized".
        func mk() -> (MTLBuffer, MTLTexture)? {
            guard let b = dev.makeBuffer(length: llen, options: .storageModeShared),
                  let t = b.makeTexture(descriptor: td, offset: 0, bytesPerRow: lbpr)
            else { return nil }
            return (b, t)
        }
        guard let (sb, st) = mk(), let (mb, mt) = mk(), let (db, dt) = mk()
        else { return nil }
        srcBuf = sb; srcTex = st; midBuf = mb; midTex = mt; dstBuf = db; dstTex = dt
    }
}

// ---- C ABI -----------------------------------------------------------------
// Coefficients are computed in PYTHON from death.py's own constants and passed
// in, so the two implementations cannot drift apart.

@_cdecl("r3dm_death_init")
public func r3dm_death_init(_ id: Int32) -> Int32 {
    guard let c = r3dmCtx(id) else { return -1 }
    if c.death != nil { return 0 }
    guard let k = DeathKit(dev: c.dev, w: c.w, h: c.h) else { return -2 }
    c.death = k
    return 0
}

@_cdecl("r3dm_death_src")
public func r3dm_death_src(_ id: Int32) -> UnsafeMutableRawPointer? {
    guard let c = r3dmCtx(id), let k = c.death else { return nil }
    return k.srcBuf.contents()
}

@_cdecl("r3dm_death_run")
public func r3dm_death_run(_ id: Int32, _ mul: Float, _ redAdd: Float,
                           _ a: Float, _ b: Float, _ cc: Float,
                           _ d: Float, _ e: Float, _ f: Float)
    -> UnsafeMutableRawPointer?
{
    guard let c = r3dmCtx(id), let k = c.death,
          let cb = c.queue.makeCommandBuffer() else { return nil }

    // pass 1: grade src -> mid
    var g = SIMD2<Float>(mul, redAdd)
    let rp1 = MTLRenderPassDescriptor()
    rp1.colorAttachments[0].texture = k.midTex
    rp1.colorAttachments[0].loadAction = .dontCare
    rp1.colorAttachments[0].storeAction = .store
    guard let e1 = cb.makeRenderCommandEncoder(descriptor: rp1) else { return nil }
    e1.setRenderPipelineState(k.grade)
    e1.setFragmentTexture(k.srcTex, index: 0)
    e1.setFragmentSamplerState(k.smp, index: 0)
    e1.setFragmentBytes(&g, length: 8, index: 0)
    e1.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    e1.endEncoding()

    // pass 2: affine warp mid -> dst
    var w = [a, b, cc, d, e, f, Float(k.w), Float(k.h)]
    let rp2 = MTLRenderPassDescriptor()
    rp2.colorAttachments[0].texture = k.dstTex
    rp2.colorAttachments[0].loadAction = .dontCare
    rp2.colorAttachments[0].storeAction = .store
    guard let e2 = cb.makeRenderCommandEncoder(descriptor: rp2) else { return nil }
    e2.setRenderPipelineState(k.warp)
    e2.setFragmentTexture(k.midTex, index: 0)
    e2.setFragmentSamplerState(k.smp, index: 0)
    e2.setFragmentBytes(&w, length: 32, index: 0)
    e2.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    e2.endEncoding()

    cb.commit()
    cb.waitUntilCompleted()
    return k.dstBuf.contents()
}
