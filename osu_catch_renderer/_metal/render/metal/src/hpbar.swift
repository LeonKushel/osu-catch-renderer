import Metal

// MSL port of the validated GLSL Argon health bar (hpgl_frag.glsl):
// 0.081 ms in-engine-legal on GL, max|diff|=3, PSNR >= 53.7 dB over 150k states.
// d_perp / t_near are STATIC geometry uploaded once as r32Float textures; the
// shader only does the per-frame endpoint clip + colour + the three composites.
let R3DM_HP_SRC = """
#include <metal_stdlib>
using namespace metal;

struct VO { float4 pos [[position]]; float2 uv; };
constant float2 Q[4] = { float2(0,0), float2(1,0), float2(0,1), float2(1,1) };

vertex VO hp_vs(uint vid [[vertex_id]]) {
    float2 q = Q[vid];
    VO o; o.pos = float4(q.x*2.0-1.0, 1.0-q.y*2.0, 0.0, 1.0); o.uv = q;
    return o;
}

// Params arrive as a FLAT float array, indexed explicitly. A struct here would
// need packed_* everywhere (MSL aligns float2/float3/float4 to 8/16/16) and one
// mistake silently misreads every field after it -- that exact bug cost a debug
// cycle in the sprite shader. 36 floats:
//  0 cw  2 m_off  4 m_size  6 g_off  8 g_size  10 m_R 11 g_R 12 m_gp 13 g_gp
// 14 hv 15 lo 16 hi  17 m_capA 19 m_capB 21 g_capA 23 g_capB
// 25 bar_rgb 28 glow_rgb 31 k_glow_rgb 34 glow_a 35 k_glow_a
#define V2(i) float2(P[i], P[i+1])
#define V3(i) float3(P[i], P[i+1], P[i+2])

static float4 colour_bar(float d, float R, float gp, float3 brgb, float ba,
                         float3 grgb, float ga) {
    float ag = R*gp;
    d = clamp(d, 0.0, R);
    float mv = clamp((R-d)/max(ag,1e-6), 0.0, 1.0);
    mv = mv*mv; mv = mv*mv; mv = mv*mv;                  // ^8
    float3 rgb = grgb; float a = ga*mv;
    if (d < R-ag) { float f = clamp((R-ag)-d, 0.0, 1.0);
        rgb = grgb*(1.0-f) + brgb*f; a = ga*(1.0-f) + ba*f; }
    if (d < R-ag-1.0) { rgb = brgb; a = ba; }
    return float4(rgb, a);
}
static float4 colour_bg(float d, float R) {
    float rel = clamp(d/R, 0.0, 1.5)/1.5;
    float3 rgb = float3(rel); float a = 0.2 + 0.6*rel;
    float rim = clamp(d-(R-1.0), 0.0, 1.0);
    if (d > R-2.0) rgb = rgb*rim + float3(1.0-rim);
    if (d > R-1.0) a = a*(1.0-rim);
    return float4(rgb, a);
}
static float restrict_d(float dp, float tn, float a, float b,
                        float2 P, float2 capA, float2 capB) {
    if (b <= a + 1e-6) return length(P - capA);
    if (tn < a) return length(P - capA);
    if (tn > b) return length(P - capB);
    return dp;
}

fragment float4 hp_fs(VO in [[stage_in]],
                      texture2d<float> frame [[texture(0)]],
                      texture2d<float> m_d   [[texture(1)]],
                      texture2d<float> m_t   [[texture(2)]],
                      texture2d<float> g_d   [[texture(3)]],
                      texture2d<float> g_t   [[texture(4)]],
                      sampler smp [[sampler(0)]],
                      constant float* P [[buffer(0)]]) {
    float2 px = float2(in.uv.x*V2(0).x, in.uv.y*V2(0).y);
    float3 col = frame.sample(smp, in.uv).rgb * 255.0;

    // 1) background track, src-over
    float2 mp = px - V2(2);
    bool inM = all(mp >= float2(0.0)) && all(mp < V2(4));
    if (inM) {
        float2 uv = mp / V2(4);
        float4 bg = colour_bg(m_d.sample(smp, uv).r, P[10]);
        col = col*(1.0-bg.a) + bg.rgb*255.0*bg.a;
    }
    // 2) glow over [lo,hi], additive
    float2 gp = px - V2(6);
    if (P[14] > 1e-4 && all(gp >= float2(0.0)) && all(gp < V2(8))) {
        float2 uv = gp / V2(8);
        float d = restrict_d(g_d.sample(smp, uv).r, g_t.sample(smp, uv).r,
                             P[15], P[16], gp, V2(21), V2(23));
        float4 gl = colour_bar(d, P[11], P[13], V3(25), 1.0,
                               V3(28), P[34]);
        float xf = gp.x / max(V2(8).x, 1.0);
        col += gl.rgb*255.0*(gl.a*(0.8 + 0.2*xf));
    }
    // 3) main bar over [0,hv], additive. NOTE the main bar's edge glow uses the
    // CONSTANT GLOW_RGB/GLOW_A, never the miss-red/heal-flash values -- getting
    // this wrong showed up as max|diff|=126 in the GL port.
    if (P[14] > 1e-4 && inM) {
        float2 uv = mp / V2(4);
        float d = restrict_d(m_d.sample(smp, uv).r, m_t.sample(smp, uv).r,
                             0.0, P[14], mp, V2(17), V2(19));
        float4 mb = colour_bar(d, P[10], P[12], float3(1.0), 1.0,
                               V3(31), P[35]);
        col += mb.rgb*255.0*mb.a;
    }
    // truncate, not round -- numpy .astype(uint8) truncates (see death.swift)
    return float4(floor(clamp(col, 0.0, 255.0))/255.0, 1.0);
}
"""

public final class HpKit {
    let pipe: MTLRenderPipelineState
    let smp: MTLSamplerState
    var sdf: [MTLTexture] = []            // m_d, m_t, g_d, g_t
    let srcBuf: MTLBuffer, dstBuf: MTLBuffer
    let srcTex: MTLTexture, dstTex: MTLTexture
    let w: Int, h: Int, bpr: Int, len: Int

    init?(dev: MTLDevice, w: Int, h: Int) {
        self.w = w; self.h = h
        let align = dev.minimumLinearTextureAlignment(for: .rgba8Unorm)
        let lbpr = ((w*4) + align - 1) / align * align
        let llen = lbpr * h
        bpr = lbpr; len = llen
        guard let lib = try? dev.makeLibrary(source: R3DM_HP_SRC, options: nil),
              let vs = lib.makeFunction(name: "hp_vs"),
              let fs = lib.makeFunction(name: "hp_fs") else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = vs; d.fragmentFunction = fs
        d.colorAttachments[0].pixelFormat = .rgba8Unorm
        guard let p = try? dev.makeRenderPipelineState(descriptor: d) else { return nil }
        pipe = p
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .nearest; sd.magFilter = .nearest   // SDF: exact texel
        sd.sAddressMode = .clampToEdge; sd.tAddressMode = .clampToEdge
        guard let s = dev.makeSamplerState(descriptor: sd) else { return nil }
        smp = s
        let td = MTLTextureDescriptor()
        td.pixelFormat = .rgba8Unorm; td.width = w; td.height = h
        td.usage = [.renderTarget, .shaderRead]; td.storageMode = .shared
        guard let sb = dev.makeBuffer(length: llen, options: .storageModeShared),
              let st = sb.makeTexture(descriptor: td, offset: 0, bytesPerRow: lbpr),
              let db = dev.makeBuffer(length: llen, options: .storageModeShared),
              let dt = db.makeTexture(descriptor: td, offset: 0, bytesPerRow: lbpr)
        else { return nil }
        srcBuf = sb; srcTex = st; dstBuf = db; dstTex = dt
    }
}

// ---- C ABI -----------------------------------------------------------------
@_cdecl("r3dm_hp_init")
public func r3dm_hp_init(_ id: Int32) -> Int32 {
    guard let c = r3dmCtx(id) else { return -1 }
    if c.hp != nil { return 0 }
    guard let k = HpKit(dev: c.dev, w: c.w, h: c.h) else { return -2 }
    c.hp = k
    return 0
}

/// Upload one static SDF plane (r32Float). `which`: 0=m_d 1=m_t 2=g_d 3=g_t.
@_cdecl("r3dm_hp_sdf")
public func r3dm_hp_sdf(_ id: Int32, _ which: Int32, _ w: Int32, _ h: Int32,
                        _ data: UnsafeRawPointer) -> Int32 {
    guard let c = r3dmCtx(id), let k = c.hp else { return -1 }
    let td = MTLTextureDescriptor()
    td.pixelFormat = .r32Float
    td.width = Int(w); td.height = Int(h)
    td.usage = [.shaderRead]; td.storageMode = .shared
    guard let t = c.dev.makeTexture(descriptor: td) else { return -2 }
    t.replace(region: MTLRegionMake2D(0, 0, Int(w), Int(h)),
              mipmapLevel: 0, withBytes: data, bytesPerRow: Int(w) * 4)
    while k.sdf.count <= Int(which) { k.sdf.append(t) }
    k.sdf[Int(which)] = t
    return 0
}

@_cdecl("r3dm_hp_src")
public func r3dm_hp_src(_ id: Int32) -> UnsafeMutableRawPointer? {
    guard let c = r3dmCtx(id), let k = c.hp else { return nil }
    return k.srcBuf.contents()
}

@_cdecl("r3dm_hp_run")
public func r3dm_hp_run(_ id: Int32, _ params: UnsafePointer<Float>)
    -> UnsafeMutableRawPointer?
{
    guard let c = r3dmCtx(id), let k = c.hp, k.sdf.count >= 4,
          let cb = c.queue.makeCommandBuffer() else { return nil }
    let rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = k.dstTex
    rp.colorAttachments[0].loadAction = .dontCare
    rp.colorAttachments[0].storeAction = .store
    guard let e = cb.makeRenderCommandEncoder(descriptor: rp) else { return nil }
    e.setRenderPipelineState(k.pipe)
    e.setFragmentTexture(k.srcTex, index: 0)
    for i in 0..<4 { e.setFragmentTexture(k.sdf[i], index: i + 1) }
    e.setFragmentSamplerState(k.smp, index: 0)
    e.setFragmentBytes(params, length: 36 * 4, index: 0)
    e.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    e.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return k.dstBuf.contents()
}
