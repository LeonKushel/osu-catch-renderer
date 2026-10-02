import Metal

// GPU Flashlight (the FL mod's vignette) -- ONE full-screen quad with MULTIPLY
// blending, drawn into the open sprite pass between the playfield sprites and
// the in-pass HUD, which is exactly where the CPU version sat (playfield
// dimmed, HUD stays lit).
//
// Why it matters: on the CPU this pass measured 9.616 ms/frame and dropped the
// renderer from 605.8 to 97.2 fps -- below the pre-Metal GL baseline. It was
// invisible for the whole optimisation session because the fixture was NoMod,
// so the `fl` stage read 0.000 ms in every perf line.
//
// No framebuffer read is needed: the fragment emits the `keep` scalar and the
// blend does dst *= keep via srcFactor = .destinationColor, dstFactor = .zero.
//
// The 800 ms size ramp is STATEFUL, so the radius is advanced on the render
// thread in frame order and passed in -- same stateless-pre-pass shape as the
// health bar. The shader itself is pure.
let R3DM_FLASH_SRC = """
#include <metal_stdlib>
using namespace metal;

struct FO { float4 pos [[position]]; float2 px; };
constant float2 FQ[4] = { float2(0,0), float2(1,0), float2(0,1), float2(1,1) };

// P: 0,1 frame w/h   2,3 centre px   4 R (lit radius)   5 Ro (outer feather)
vertex FO flash_vs(uint vid [[vertex_id]], constant float* P [[buffer(0)]]) {
    float2 q = FQ[vid];
    FO o;
    o.pos = float4(q.x * 2.0 - 1.0, 1.0 - q.y * 2.0, 0.0, 1.0);
    o.px  = q * float2(P[0], P[1]);
    return o;
}

fragment float4 flash_fs(FO in [[stage_in]], constant float* P [[buffer(0)]]) {
    // flashlight.py evaluates on INTEGER pixel indices (np.arange), so sample
    // at the pixel index, not the interpolated centre -- the same half-pixel
    // correction the progress pill needed.
    float2 d = (in.px - 0.5) - float2(P[2], P[3]);
    float dist = sqrt(d.x * d.x + d.y * d.y);
    float R = P[4], Ro = P[5];
    float t = clamp((dist - R) / max(Ro - R, 1e-6), 0.0, 1.0);
    float black = t * t * (3.0 - 2.0 * t);          // smoothstep, as in numpy
    float keep = 1.0 - black;
    return float4(keep, keep, keep, 1.0);
}
""";

final class FlashKit {
    let mul: MTLRenderPipelineState
    init?(dev: MTLDevice, fmt: MTLPixelFormat) {
        guard let lib = try? dev.makeLibrary(source: R3DM_FLASH_SRC, options: nil),
              let vs = lib.makeFunction(name: "flash_vs"),
              let fs = lib.makeFunction(name: "flash_fs") else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = vs; d.fragmentFunction = fs
        let a = d.colorAttachments[0]!
        a.pixelFormat = fmt
        a.isBlendingEnabled = true
        // dst = src * dst  -- a straight multiply. Alpha is left alone.
        a.sourceRGBBlendFactor = .destinationColor
        a.destinationRGBBlendFactor = .zero
        a.sourceAlphaBlendFactor = .zero
        a.destinationAlphaBlendFactor = .one
        guard let p = try? dev.makeRenderPipelineState(descriptor: d) else { return nil }
        mul = p
    }
}

/// Multiply the open pass by the flashlight vignette. 6 floats, see flash_vs.
@_cdecl("r3dm_flash_inline")
public func r3dm_flash_inline(_ id: Int32, _ params: UnsafePointer<Float>) -> Int32 {
    guard let c = r3dmCtx(id), let e = c.enc else { return -1 }
    if c.flash == nil {
        guard let k = FlashKit(dev: c.dev, fmt: .rgba8Unorm) else { return -2 }
        c.flash = k
    }
    e.setRenderPipelineState(c.flash!.mul)
    e.setVertexBytes(params, length: 6 * 4, index: 0)
    e.setFragmentBytes(params, length: 6 * 4, index: 0)
    e.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
    // r3dm_draw re-binds the viewport itself now, so no restore needed here --
    // see the comment on that function about the nil'd `screen` uniform.
    return 0
}
