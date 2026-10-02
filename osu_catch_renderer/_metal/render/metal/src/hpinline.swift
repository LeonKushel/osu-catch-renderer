import Metal

// In-pass health bar: THREE blended draws into the sprite pass's own render
// target. No round trip and no frame sampling -- the bar's layers ARE blend
// modes (bg src-over, glow additive, main additive), which is how the GL path
// does it and was validated there at max|diff|=3 / PSNR >= 54 dB.
// Params: the same flat 36-float array as hp_fs (see hpbar.swift).
let R3DM_HPI_SRC = """
#include <metal_stdlib>
using namespace metal;

struct VO { float4 pos [[position]]; float2 uv; };

#define V2(i) float2(P[i], P[i+1])
#define V3(i) float3(P[i], P[i+1], P[i+2])

// Quad covering just this layer's rect, in frame pixels -> NDC (top-left origin)
vertex VO hpi_vs(uint vid [[vertex_id]],
                 constant float* P [[buffer(0)]],
                 constant int& mode [[buffer(1)]]) {
    float2 off  = (mode == 1) ? V2(6) : V2(2);      // glow uses g_off/g_size
    float2 size = (mode == 1) ? V2(8) : V2(4);
    float2 cw   = V2(0);
    float2 q = float2(float(vid & 1), float(vid >> 1));
    float2 px = off + q * size;
    VO o;
    o.pos = float4(px.x / cw.x * 2.0 - 1.0, 1.0 - px.y / cw.y * 2.0, 0.0, 1.0);
    o.uv  = q;
    return o;
}

static float4 cbar(float d, float R, float gp, float3 brgb, float ba,
                   float3 grgb, float ga) {
    float ag = R*gp; d = clamp(d, 0.0, R);
    float mv = clamp((R-d)/max(ag,1e-6), 0.0, 1.0);
    mv = mv*mv; mv = mv*mv; mv = mv*mv;
    float3 rgb = grgb; float a = ga*mv;
    if (d < R-ag) { float f = clamp((R-ag)-d, 0.0, 1.0);
        rgb = grgb*(1.0-f) + brgb*f; a = ga*(1.0-f) + ba*f; }
    if (d < R-ag-1.0) { rgb = brgb; a = ba; }
    return float4(rgb, a);
}
static float4 cbg(float d, float R) {
    float rel = clamp(d/R, 0.0, 1.5)/1.5;
    float3 rgb = float3(rel); float a = 0.2 + 0.6*rel;
    float rim = clamp(d-(R-1.0), 0.0, 1.0);
    if (d > R-2.0) rgb = rgb*rim + float3(1.0-rim);
    if (d > R-1.0) a = a*(1.0-rim);
    return float4(rgb, a);
}
static float rd(float dp, float tn, float a, float b, float2 Pp,
                float2 cA, float2 cB) {
    if (b <= a + 1e-6) return length(Pp - cA);
    if (tn < a) return length(Pp - cA);
    if (tn > b) return length(Pp - cB);
    return dp;
}

fragment float4 hpi_fs(VO in [[stage_in]],
                       texture2d<float> m_d [[texture(0)]],
                       texture2d<float> m_t [[texture(1)]],
                       texture2d<float> g_d [[texture(2)]],
                       texture2d<float> g_t [[texture(3)]],
                       sampler smp [[sampler(0)]],
                       constant float* P [[buffer(0)]],
                       constant int& mode [[buffer(1)]]) {
    if (mode == 0) {                       // background track, src-over
        return cbg(m_d.sample(smp, in.uv).r, P[10]);
    }
    if (mode == 1) {                       // glow, additive, [lo,hi]
        float2 gp = in.uv * V2(8);
        float d = rd(g_d.sample(smp, in.uv).r, g_t.sample(smp, in.uv).r,
                     P[15], P[16], gp, V2(21), V2(23));
        float4 c = cbar(d, P[11], P[13], V3(25), 1.0, V3(28), P[34]);
        c.a *= (0.8 + 0.2 * in.uv.x);      // horizontal gradient
        return c;
    }
    // mode 2: main bar, additive, [0,hv]. Constant GLOW_RGB/GLOW_A for the edge,
    // never the miss/flash values -- that mistake read max|diff|=126 on GL.
    float2 mp = in.uv * V2(4);
    float d = rd(m_d.sample(smp, in.uv).r, m_t.sample(smp, in.uv).r,
                 0.0, P[14], mp, V2(17), V2(19));
    return cbar(d, P[10], P[12], float3(1.0), 1.0, V3(31), P[35]);
}
"""

public final class HpInline {
    let bg: MTLRenderPipelineState
    let add: MTLRenderPipelineState
    let smp: MTLSamplerState
    init?(dev: MTLDevice, fmt: MTLPixelFormat) {
        guard let lib = try? dev.makeLibrary(source: R3DM_HPI_SRC, options: nil),
              let vs = lib.makeFunction(name: "hpi_vs"),
              let fs = lib.makeFunction(name: "hpi_fs") else { return nil }
        func pipe(_ additive: Bool) -> MTLRenderPipelineState? {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = vs; d.fragmentFunction = fs
            let a = d.colorAttachments[0]!
            a.pixelFormat = fmt
            a.isBlendingEnabled = true
            a.rgbBlendOperation = .add; a.alphaBlendOperation = .add
            a.sourceRGBBlendFactor = .sourceAlpha
            a.sourceAlphaBlendFactor = .sourceAlpha
            a.destinationRGBBlendFactor = additive ? .one : .oneMinusSourceAlpha
            a.destinationAlphaBlendFactor = additive ? .one : .oneMinusSourceAlpha
            return try? dev.makeRenderPipelineState(descriptor: d)
        }
        guard let p0 = pipe(false), let p1 = pipe(true) else { return nil }
        bg = p0; add = p1
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .nearest; sd.magFilter = .nearest
        sd.sAddressMode = .clampToEdge; sd.tAddressMode = .clampToEdge
        guard let s = dev.makeSamplerState(descriptor: sd) else { return nil }
        smp = s
    }
}

/// Draw the bar into the OPEN sprite-pass encoder. Must be called between
/// r3dm_begin and r3dm_commit, after the scene sprites.
@_cdecl("r3dm_hp_inline")
public func r3dm_hp_inline(_ id: Int32, _ params: UnsafePointer<Float>) -> Int32 {
    guard let c = r3dmCtx(id), let e = c.enc, let k = c.hp, k.sdf.count >= 4
    else { return -1 }
    if c.hpi == nil {
        guard let h = HpInline(dev: c.dev, fmt: .rgba8Unorm) else { return -2 }
        c.hpi = h
    }
    let h = c.hpi!
    let hv = params[14]
    e.setVertexBytes(params, length: 36 * 4, index: 0)
    e.setFragmentBytes(params, length: 36 * 4, index: 0)
    for i in 0..<4 { e.setFragmentTexture(k.sdf[i], index: i) }
    e.setFragmentSamplerState(h.smp, index: 0)
    var modes: [Int32] = hv > 1e-4 ? [0, 1, 2] : [0]
    for m in modes {
        var mm = m
        e.setRenderPipelineState(m == 0 ? h.bg : h.add)
        e.setVertexBytes(&mm, length: 4, index: 1)
        e.setFragmentBytes(&mm, length: 4, index: 1)
        e.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    }
    // Restore the viewport this shader borrowed at index 1. Setting it to nil
    // (what this line used to do) left later sprite draws reading a nil
    // `screen` -- they collapsed to zero-size quads and vanished.
    var screen = SIMD2<Float>(Float(c.w), Float(c.h))
    e.setVertexBytes(&screen, length: 8, index: 1)
    return 0
}
