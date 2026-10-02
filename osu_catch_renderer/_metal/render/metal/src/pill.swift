import Metal

// Rounded-pill SDF, an exact port of argon_hud.bake_pill_alpha:
//   r  = h/2 - 1
//   qx = |x - (w-1)/2| - (w/2 - 1 - r)
//   qy = |y - (h-1)/2| - (h/2 - 1 - r)
//   d  = hypot(max(qx,0), max(qy,0)) + min(max(qx,qy), 0) - r
//   a  = clip(-d / AA, 0, 1)
// Needed as a shader (not a baked texture) only for the progress FILL pill,
// whose width tracks playback and so would need one bake per pixel of width.
let R3DM_PILL_SRC = """
#include <metal_stdlib>
using namespace metal;

struct VO { float4 pos [[position]]; float2 px; };

// params: 0,1 frame w/h  2,3 rect origin  4,5 rect size (= pill w/h)
//         6 aa  7 alpha_mul  8,9,10 rgb(0..1)
vertex VO pill_vs(uint vid [[vertex_id]], constant float* P [[buffer(0)]]) {
    float2 q = float2(float(vid & 1), float(vid >> 1));
    float2 p = float2(P[2], P[3]) + q * float2(P[4], P[5]);
    VO o;
    o.pos = float4(p.x / P[0] * 2.0 - 1.0, 1.0 - p.y / P[1] * 2.0, 0.0, 1.0);
    o.px  = q * float2(P[4], P[5]);        // pixel coords inside the pill
    return o;
}

fragment float4 pill_fs(VO in [[stage_in]], constant float* P [[buffer(0)]]) {
    float w = P[4], h = P[5];
    // bake_pill_alpha evaluates on INTEGER pixel indices (np.mgrid), but the
    // interpolated quad coordinate lands on pixel CENTRES (i + 0.5). Without
    // this -0.5 the AA band is off by half a pixel: max|diff| was 120/255.
    float2 px = in.px - 0.5;
    float r = h * 0.5 - 1.0;
    float qx = abs(px.x - (w - 1.0) * 0.5) - (w * 0.5 - 1.0 - r);
    float qy = abs(px.y - (h - 1.0) * 0.5) - (h * 0.5 - 1.0 - r);
    float d = length(float2(max(qx, 0.0), max(qy, 0.0)))
              + min(max(qx, qy), 0.0) - r;
    float a = clamp(-d / P[6], 0.0, 1.0) * P[7];
    return float4(P[8], P[9], P[10], a);
}
"""

public final class PillKit {
    let over: MTLRenderPipelineState
    init?(dev: MTLDevice, fmt: MTLPixelFormat) {
        guard let lib = try? dev.makeLibrary(source: R3DM_PILL_SRC, options: nil),
              let vs = lib.makeFunction(name: "pill_vs"),
              let fs = lib.makeFunction(name: "pill_fs") else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = vs; d.fragmentFunction = fs
        let a = d.colorAttachments[0]!
        a.pixelFormat = fmt
        a.isBlendingEnabled = true
        a.rgbBlendOperation = .add; a.alphaBlendOperation = .add
        a.sourceRGBBlendFactor = .sourceAlpha
        a.sourceAlphaBlendFactor = .sourceAlpha
        a.destinationRGBBlendFactor = .oneMinusSourceAlpha
        a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        guard let p = try? dev.makeRenderPipelineState(descriptor: d) else { return nil }
        over = p
    }
}

/// Draw one src-over pill into the OPEN sprite pass. 11 floats, see pill_vs.
@_cdecl("r3dm_pill_inline")
public func r3dm_pill_inline(_ id: Int32, _ params: UnsafePointer<Float>) -> Int32 {
    guard let c = r3dmCtx(id), let e = c.enc else { return -1 }
    if c.pill == nil {
        guard let k = PillKit(dev: c.dev, fmt: .rgba8Unorm) else { return -2 }
        c.pill = k
    }
    e.setRenderPipelineState(c.pill!.over)
    e.setVertexBytes(params, length: 11 * 4, index: 0)
    e.setFragmentBytes(params, length: 11 * 4, index: 0)
    e.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    return 0
}
