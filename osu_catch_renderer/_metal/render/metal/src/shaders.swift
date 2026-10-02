import Metal

// MSL port of the GL instanced sprite path (gl.py _I_VERT / _I_FRAG).
// Same 14-float per-instance layout, same 16 texture units, same flat-int
// selection, same painter's order within a blend group.
//
// NOTE on orientation: Metal's framebuffer origin is TOP-LEFT while GL's is
// bottom-left, so the identical vertex math yields a top-down image -- which is
// what the encoder wants. That deletes the GL path's extra flip render pass.
let R3DM_SHADER_SRC = """
#include <metal_stdlib>
using namespace metal;

// PACKED types are mandatory. MSL aligns float4 to 16 bytes, so a plain
// `float4 color` after `float rot` (offset 16) is read from offset 32 and every
// field past it is misaligned against the engine's tightly-packed 14-float
// instance array. packed_* have 4-byte alignment: 8+8+4+16+8+8+4 = 56 = 14 floats.
// (Caught by a blend test: a green instance rendered pure red.)
struct Inst {
    packed_float2 center;
    packed_float2 size;
    float         rot;
    packed_float4 color;
    packed_float2 uv_off;
    packed_float2 uv_scale;
    float         tex;
};
static_assert(sizeof(Inst) == 56, "Inst must be 14 tightly-packed floats");

struct VOut {
    float4 pos [[position]];
    float2 uv;
    float4 color;
    uint   tex [[flat]];
};

// unit quad as a triangle strip: (0,0) (1,0) (0,1) (1,1)
constant float2 QUAD[4] = { float2(0,0), float2(1,0), float2(0,1), float2(1,1) };

vertex VOut r3dm_vs(uint vid [[vertex_id]],
                    uint iid [[instance_id]],
                    constant Inst*  insts   [[buffer(0)]],
                    constant float2& screen [[buffer(1)]])
{
    Inst I = insts[iid];
    float2 q = QUAD[vid];
    float2 icenter = float2(I.center);
    float2 isize   = float2(I.size);
    float2 iuvoff  = float2(I.uv_off);
    float2 iuvsc   = float2(I.uv_scale);
    float2 local = (q - 0.5) * isize;            // centre-origin quad
    float  c = cos(I.rot), s = sin(I.rot);
    float2 rotd = float2(local.x * c - local.y * s,
                         local.x * s + local.y * c);
    float2 px = icenter + rotd;                  // pixel space, top-left origin
    float2 ndc = float2(px.x / screen.x * 2.0 - 1.0,
                        1.0 - px.y / screen.y * 2.0);
    VOut o;
    o.pos   = float4(ndc, 0.0, 1.0);
    o.uv    = q * iuvsc + iuvoff;
    o.color = float4(I.color);
    o.tex   = uint(I.tex + 0.5);
    return o;
}

fragment float4 r3dm_fs(VOut in [[stage_in]],
                        array<texture2d<float>, 16> tex [[texture(0)]],
                        sampler smp [[sampler(0)]])
{
    float4 t = tex[in.tex].sample(smp, in.uv);
    return t * in.color;
}
"""

func r3dmMakePipelines(_ dev: MTLDevice, _ fmt: MTLPixelFormat)
    -> (MTLRenderPipelineState, MTLRenderPipelineState, MTLSamplerState)?
{
    guard let lib = try? dev.makeLibrary(source: R3DM_SHADER_SRC, options: nil),
          let vs = lib.makeFunction(name: "r3dm_vs"),
          let fs = lib.makeFunction(name: "r3dm_fs") else { return nil }

    func pipe(_ additive: Bool) -> MTLRenderPipelineState? {
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = vs; d.fragmentFunction = fs
        let a = d.colorAttachments[0]!
        a.pixelFormat = fmt
        a.isBlendingEnabled = true
        a.rgbBlendOperation = .add
        a.alphaBlendOperation = .add
        a.sourceRGBBlendFactor = .sourceAlpha
        a.sourceAlphaBlendFactor = .sourceAlpha
        // GL: normal = (SRC_ALPHA, ONE_MINUS_SRC_ALPHA); additive = (SRC_ALPHA, ONE)
        a.destinationRGBBlendFactor = additive ? .one : .oneMinusSourceAlpha
        a.destinationAlphaBlendFactor = additive ? .one : .oneMinusSourceAlpha
        return try? dev.makeRenderPipelineState(descriptor: d)
    }
    guard let pn = pipe(false), let pa = pipe(true) else { return nil }

    let sd = MTLSamplerDescriptor()
    sd.minFilter = .linear; sd.magFilter = .linear
    sd.mipFilter = .linear          // match GL's mipmapped LINEAR
    sd.sAddressMode = .clampToEdge; sd.tAddressMode = .clampToEdge
    guard let smp = dev.makeSamplerState(descriptor: sd) else { return nil }
    return (pn, pa, smp)
}
