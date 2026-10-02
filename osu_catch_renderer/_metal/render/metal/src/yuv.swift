import Metal

// RGBA8 -> yuv420p on the GPU, so ffmpeg receives 3.11 MB of planar YUV
// instead of 8.29 MB of RGBA and runs NO swscale at all.
//
// This is a port of swscale's own path, not a fresh conversion, so the output
// tracks what the pipeline produces today. That path was recovered by fitting
// swscale's output (ffmpeg 8.1, default flags) and is NOT what you'd guess:
//
//   Y        per pixel, BT.601 limited range, RGB2YUV_SHIFT = 15
//   chroma   HORIZONTAL: a plain 2-pixel SUM in RGB, then convert
//            VERTICAL:   an 8-tap BICUBIC over the half-width chroma rows
//
// A 2x2 box average -- the obvious implementation -- is wrong: it matches
// `-sws_flags area`, not the default, and lands 5-6 LSB out on real frames.
// Measured against ffmpeg's default on 8 real frames: Y 100% bit-exact,
// chroma 99.8% exact, worst error 1 LSB (swscale's final rounding bit, which
// needs its source to reproduce -- and would pin us to one ffmpeg version).
//
// Integers throughout, read straight out of a shared buffer: sampling the
// frame as a texture would round the 8-bit values before we ever got them.
//
// RINGED AND PIPELINED. One input buffer plus waitUntilCompleted cost ~1.0 ms
// per frame on the composite thread -- the binding stage -- because the thread
// stalled on GPU latency every frame. Dropping the wait REQUIRES a ring: with
// a single input buffer the next frame's write would race the in-flight kernel
// reading it. Same lesson as the sprite ring, and the same lesson the death
// path taught the hard way -- a reused buffer handed to a queued consumer is a
// corruption bug that shows up as nondeterminism, not as a crash.
let R3DM_YUV_SRC = """
#include <metal_stdlib>
using namespace metal;

// swscale input.c coefficients at RGB2YUV_SHIFT = 15, BT.601 limited range
constant int RY =  8414, GY =  16519, BY =  3208;
constant int RU = -4865, GU =  -9528, BU = 14392;
constant int RV = 14392, GV = -12061, BV = -2332;
// the 2:1 vertical bicubic, 8 taps. Sums to 256 because the chroma rows are
// carried at 64x the 8-bit scale (that is what makes the >> 14 come out right).
constant int VT[8] = { -4, -11, 31, 112, 112, 31, -11, -4 };

struct Geom { int w; int h; int bpr; };

// Reads the render target through an rgba8Uint VIEW. Same arithmetic as the
// buffer version -- texture reads of an integer view are exact, no filtering,
// no normalisation -- but Metal tracks the dependency on the render pass, so
// the kernel cannot run before the pixels land.
kernel void tex_to_yuv420p(texture2d<uint, access::read> src [[texture(0)]],
                           device uchar *dst [[buffer(1)]],
                           constant Geom& g  [[buffer(2)]],
                           uint2 gid [[thread_position_in_grid]])
{
    int cw = g.w >> 1, ch = g.h >> 1;
    if (int(gid.x) >= cw || int(gid.y) >= ch) return;
    int x = int(gid.x) * 2, y = int(gid.y) * 2;
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            uint4 p = src.read(uint2(x + dx, y + dy));
            int R = int(p.r), G = int(p.g), B = int(p.b);
            int Y = ((((RY * R + GY * G + BY * B) + (0x801 << 8)) >> 9) + 32) >> 6;
            dst[(y + dy) * g.w + (x + dx)] = uchar(clamp(Y, 0, 255));
        }
    }
    int accU = 0, accV = 0;
    for (int k = 0; k < 8; ++k) {
        int sy = clamp(y + k - 3, 0, g.h - 1);
        uint4 p0 = src.read(uint2(x, sy));
        uint4 p1 = src.read(uint2(x + 1, sy));
        int R = int(p0.r) + int(p1.r);
        int G = int(p0.g) + int(p1.g);
        int B = int(p0.b) + int(p1.b);
        accU += VT[k] * (((RU * R + GU * G + BU * B) + (0x4001 << 9)) >> 10);
        accV += VT[k] * (((RV * R + GV * G + BV * B) + (0x4001 << 9)) >> 10);
    }
    int uoff = g.w * g.h;
    int voff = uoff + cw * ch;
    int ci = int(gid.y) * cw + int(gid.x);
    dst[uoff + ci] = uchar(clamp((accU + 8192) >> 14, 0, 255));
    dst[voff + ci] = uchar(clamp((accV + 8192) >> 14, 0, 255));
}

kernel void rgba_to_yuv420p(device const uchar *src [[buffer(0)]],
                            device uchar       *dst [[buffer(1)]],
                            constant Geom&      g   [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    int cw = g.w >> 1, ch = g.h >> 1;
    if (int(gid.x) >= cw || int(gid.y) >= ch) return;
    int x = int(gid.x) * 2, y = int(gid.y) * 2;

    // ---- Y: the 2x2 quad this thread owns, each pixel on its own
    for (int dy = 0; dy < 2; ++dy) {
        for (int dx = 0; dx < 2; ++dx) {
            int o = (y + dy) * g.bpr + (x + dx) * 4;
            int R = int(src[o]), G = int(src[o + 1]), B = int(src[o + 2]);
            int Y = ((((RY * R + GY * G + BY * B) + (0x801 << 8)) >> 9) + 32) >> 6;
            dst[(y + dy) * g.w + (x + dx)] = uchar(clamp(Y, 0, 255));
        }
    }

    // ---- U/V: 2-pixel horizontal sum per row, 8-tap bicubic down the rows
    int accU = 0, accV = 0;
    for (int k = 0; k < 8; ++k) {
        int sy = clamp(y + k - 3, 0, g.h - 1);      // swscale clamps at the edges
        int o = sy * g.bpr + x * 4;
        int R = int(src[o])     + int(src[o + 4]);
        int G = int(src[o + 1]) + int(src[o + 5]);
        int B = int(src[o + 2]) + int(src[o + 6]);
        accU += VT[k] * (((RU * R + GU * G + BU * B) + (0x4001 << 9)) >> 10);
        accV += VT[k] * (((RV * R + GV * G + BV * B) + (0x4001 << 9)) >> 10);
    }
    int uoff = g.w * g.h;
    int voff = uoff + cw * ch;
    int ci = int(gid.y) * cw + int(gid.x);
    dst[uoff + ci] = uchar(clamp((accU + 8192) >> 14, 0, 255));
    dst[voff + ci] = uchar(clamp((accV + 8192) >> 14, 0, 255));
}
""";

final class YuvKit {
    let pipe: MTLComputePipelineState
    var texPipe: MTLComputePipelineState?
    let inBufs: [MTLBuffer]              // RGBA in,  one per ring slot
    let outBufs: [MTLBuffer]             // yuv out,  one per ring slot
    var inflight: [MTLCommandBuffer?]
    var head = 0, tail = 0               // submitted / consumed
    let ring: Int, depth: Int
    let w: Int, h: Int, bpr: Int, outLen: Int

    init?(dev: MTLDevice, w: Int, h: Int, ring: Int, depth: Int) {
        guard let lib = try? dev.makeLibrary(source: R3DM_YUV_SRC, options: nil),
              let fn = lib.makeFunction(name: "rgba_to_yuv420p"),
              let p = try? dev.makeComputePipelineState(function: fn)
        else { return nil }
        self.w = w; self.h = h; self.bpr = w * 4
        self.outLen = w * h + 2 * ((w / 2) * (h / 2))
        self.ring = max(2, ring); self.depth = max(1, min(depth, max(2, ring) - 1))
        var ib: [MTLBuffer] = [], ob: [MTLBuffer] = []
        for _ in 0..<self.ring {
            guard let a = dev.makeBuffer(length: bpr * h, options: .storageModeShared),
                  let b = dev.makeBuffer(length: outLen, options: .storageModeShared)
            else { return nil }
            ib.append(a); ob.append(b)
        }
        inBufs = ib; outBufs = ob
        if let fn2 = lib.makeFunction(name: "tex_to_yuv420p") {
            texPipe = try? dev.makeComputePipelineState(function: fn2)
        }
        inflight = [MTLCommandBuffer?](repeating: nil, count: self.ring)
        pipe = p
    }
}

@_cdecl("r3dm_yuv_init")
public func r3dm_yuv_init(_ id: Int32, _ ring: Int32, _ depth: Int32) -> Int32 {
    guard let c = r3dmCtx(id) else { return -1 }
    if c.yuv != nil { return 0 }
    guard let k = YuvKit(dev: c.dev, w: c.w, h: c.h,
                         ring: Int(ring), depth: Int(depth)) else { return -2 }
    c.yuv = k
    return 0
}

@_cdecl("r3dm_yuv_len")
public func r3dm_yuv_len(_ id: Int32) -> Int32 {
    guard let c = r3dmCtx(id), let k = c.yuv else { return -1 }
    return Int32(k.outLen)
}

/// Pointer to the NEXT input slot to fill, or NULL while the ring is full.
/// Write the composited RGBA frame here, then call submit.
@_cdecl("r3dm_yuv_input")
public func r3dm_yuv_input(_ id: Int32) -> UnsafeMutableRawPointer? {
    guard let c = r3dmCtx(id), let k = c.yuv else { return nil }
    if k.head - k.tail >= k.ring { return nil }
    return k.inBufs[k.head % k.ring].contents()
}

/// Dispatch the slot that r3dm_yuv_input just handed out. Does NOT wait.
@_cdecl("r3dm_yuv_submit")
public func r3dm_yuv_submit(_ id: Int32) -> Int32 {
    guard let c = r3dmCtx(id), let k = c.yuv else { return -1 }
    if k.head - k.tail >= k.ring { return -2 }
    let s = k.head % k.ring
    guard let cb = c.queue.makeCommandBuffer(),
          let e = cb.makeComputeCommandEncoder() else { return -3 }
    var g = (Int32(k.w), Int32(k.h), Int32(k.bpr))
    e.setComputePipelineState(k.pipe)
    e.setBuffer(k.inBufs[s], offset: 0, index: 0)
    e.setBuffer(k.outBufs[s], offset: 0, index: 1)
    withUnsafeBytes(of: &g) { e.setBytes($0.baseAddress!, length: 12, index: 2) }
    let tw = k.pipe.threadExecutionWidth
    let th = max(1, k.pipe.maxTotalThreadsPerThreadgroup / tw)
    e.dispatchThreads(MTLSize(width: k.w / 2, height: k.h / 2, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    e.endEncoding()
    cb.commit()
    k.inflight[s] = cb
    k.head += 1
    return 0
}

/// Oldest completed conversion, or NULL while the pipeline is still filling.
/// `force` drains at end of stream. The returned pointer is valid until this
/// slot comes round again -- `ring` frames later -- so the caller must either
/// copy it or guarantee the consumer is shallower than the ring.
@_cdecl("r3dm_yuv_acquire")
public func r3dm_yuv_acquire(_ id: Int32, _ force: Int32) -> UnsafeMutableRawPointer? {
    guard let c = r3dmCtx(id), let k = c.yuv else { return nil }
    if k.head == k.tail { return nil }
    if force == 0 && k.head - k.tail < k.depth { return nil }   // keep it pipelined
    let s = k.tail % k.ring
    k.inflight[s]?.waitUntilCompleted()
    k.inflight[s] = nil
    k.tail += 1
    return k.outBufs[s].contents()
}

/// Convert the OLDEST finished render-target slot straight to yuv420p, with no
/// readback and no host copy of the RGBA at all.
///
/// The normal path costs 8.29 MB out of the ring + 8.29 MB into this kit's
/// input buffer + 3.11 MB out = 19.7 MB moved per frame. The ring's render
/// targets are buffer-backed (`bufs[k]`, padded `bpr`), so the kernel can read
/// the pass's own bytes and only the 3.11 MB result ever crosses to the host.
///
/// ONLY valid when the frame is COMPLETE in the render target -- i.e. nothing
/// is drawn on the CPU afterwards. True for GPU-composed outro frames; NOT true
/// for gameplay frames while any HUD element is still PIL-side.
@_cdecl("r3dm_yuv_from_ring")
public func r3dm_yuv_from_ring(_ id: Int32) -> UnsafeMutableRawPointer? {
    guard let c = r3dmCtx(id), let k = c.yuv, c.tail < c.head else { return nil }
    let s = c.tail % c.ring
    c.inflight[s]?.waitUntilCompleted()        // the render pass for this slot
    c.inflight[s] = nil
    guard let cb = c.queue.makeCommandBuffer(),
          let e = cb.makeComputeCommandEncoder() else { return nil }
    var g = (Int32(k.w), Int32(k.h), Int32(c.bpr))    // the RING's row stride
    e.setComputePipelineState(k.pipe)
    e.setBuffer(c.bufs[s], offset: 0, index: 0)
    e.setBuffer(k.outBufs[0], offset: 0, index: 1)
    withUnsafeBytes(of: &g) { e.setBytes($0.baseAddress!, length: 12, index: 2) }
    let tw = k.pipe.threadExecutionWidth
    let th = max(1, k.pipe.maxTotalThreadsPerThreadgroup / tw)
    e.dispatchThreads(MTLSize(width: k.w / 2, height: k.h / 2, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    c.tail += 1
    return k.outBufs[0].contents()
}

/// Commit the open render pass AND chain the yuv420p conversion into the SAME
/// command buffer, then return immediately.
///
/// This is the whole point: the previous version waited for the render pass,
/// dispatched the kernel, and waited again -- two serialised GPU round trips
/// per frame on a single thread, which lost to a 3-worker CPU compositor even
/// after the host copies were removed. Here the CPU issues work and moves on;
/// finished frames are collected by r3dm_yuv_ring_acquire.
///
/// Both encoders are in one command buffer, so Metal orders the compute after
/// the render and the kernel sees the finished pixels.
@_cdecl("r3dm_commit_yuv")
public func r3dm_commit_yuv(_ id: Int32) -> Int32 {
    guard let c = r3dmCtx(id), let k = c.yuv, let cb = c.cur else { return -1 }
    c.enc?.endEncoding(); c.enc = nil
    let s = c.head % c.ring
    if c.yuvRing.isEmpty {
        for _ in 0..<c.ring {
            guard let b = c.dev.makeBuffer(length: k.outLen,
                                           options: .storageModeShared)
            else { return -2 }
            c.yuvRing.append(b)
        }
    }
    if c.texesUint.isEmpty {
        for t in c.texes {
            guard let v = t.makeTextureView(pixelFormat: .rgba8Uint)
            else { return -4 }
            c.texesUint.append(v)
        }
    }
    guard let tp = k.texPipe,
          let e = cb.makeComputeCommandEncoder() else { return -3 }
    var g = (Int32(k.w), Int32(k.h), Int32(c.bpr))
    e.setComputePipelineState(tp)
    e.setTexture(c.texesUint[s], index: 0)
    e.setBuffer(c.yuvRing[s], offset: 0, index: 1)
    withUnsafeBytes(of: &g) { e.setBytes($0.baseAddress!, length: 12, index: 2) }
    let tw = tp.threadExecutionWidth
    let th = max(1, tp.maxTotalThreadsPerThreadgroup / tw)
    e.dispatchThreads(MTLSize(width: k.w / 2, height: k.h / 2, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: tw, height: th, depth: 1))
    e.endEncoding()
    cb.commit()                      // NO waitUntilCompleted
    c.inflight[s] = cb
    c.head += 1
    c.cur = nil
    return 0
}

/// Oldest finished yuv420p frame, or NULL while the pipeline fills.
/// `force` drains. Only the 3.11 MB result ever crosses to the host.
@_cdecl("r3dm_yuv_ring_acquire")
public func r3dm_yuv_ring_acquire(_ id: Int32, _ depth: Int32,
                                  _ force: Int32) -> UnsafeMutableRawPointer? {
    guard let c = r3dmCtx(id), c.tail < c.head, !c.yuvRing.isEmpty else {
        return nil
    }
    if force == 0 && (c.head - c.tail) < Int(depth) { return nil }
    let s = c.tail % c.ring
    c.inflight[s]?.waitUntilCompleted()
    c.inflight[s] = nil
    c.tail += 1
    return c.yuvRing[s].contents()
}
