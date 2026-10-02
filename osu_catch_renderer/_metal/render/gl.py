"""Minimal moderngl sprite batch for the catch renderer.

Owns a standalone EGL context and an offscreen RGBA framebuffer. Draws
textured/solid quads with straight-alpha blending in painter's order, then a
GPU flip pass + RGBA readback feeds the zero-copy CPU compositing pipeline
(ffmpeg ingests -pix_fmt rgba; alpha is ignored). Deliberately tiny and
self-contained so it can be discarded when the VRender branch takes over.
"""
from __future__ import annotations

import sys
import os

import numpy as np

try:
    import moderngl
except Exception as e:  # noqa: BLE001
    raise RuntimeError("moderngl is required for the catch renderer") from e

from osu_catch_renderer._metal.beatmap.models import Sprite

# Draw-path instrumentation, all resolved at import so the hot loop is unchanged
# when they are off. R3D_DRAW_COUNT=1 counts sprites; R3D_DRAW_NORENDER=1 keeps
# every uniform write but skips vao.render(), isolating Python/uniform overhead
# from the GL draw-call + fill cost.
_DRAW_COUNT = os.environ.get("R3D_DRAW_COUNT") == "1"
# R3D_DRAW_FINISH=1 forces a GPU sync at the END of draw(). Diagnostic only:
# GL draw calls are async, so draw() normally measures CPU issuance and the
# GPU's actual execution is paid later, inside read_rgb_async's readback. This
# moves that wait into `draw` so the split is visible.
_DRAW_FINISH = os.environ.get("R3D_DRAW_FINISH") == "1"
_INSTANCED = os.environ.get("R3D_INSTANCED_DRAW") == "1"
_DRAW_NORENDER = os.environ.get("R3D_DRAW_NORENDER") == "1"
_DRAW_STATS = [0, 0, 0, 0, 0, 0, 0, 0]  # +runs, max runs

_VERT = """
#version 330
in vec2 in_pos;      // unit quad corner [-0.5,0.5]
in vec2 in_uv;
uniform vec2 u_screen;   // (w, h) in px
uniform vec2 u_center;   // sprite center in px (origin top-left)
uniform vec2 u_size;     // sprite w,h in px
uniform float u_rot;     // radians
uniform vec2 u_uv_off;   // texture UV offset (storyboard flip mirroring)
uniform vec2 u_uv_scale; // texture UV scale  (default (1,1); -1 mirrors an axis)
out vec2 v_uv;
void main() {
    vec2 p = in_pos * u_size;
    float c = cos(u_rot), s = sin(u_rot);
    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    vec2 px = u_center + p;
    // px -> clip, with y flipped (top-left origin)
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    // identity default ((0,0)/(1,1)) == `in_uv`, so non-storyboard sprites are
    // sampled bit-identically; a storyboard flip passes off=1,scale=-1 per axis.
    v_uv = in_uv * u_uv_scale + u_uv_off;
}
"""

_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
uniform vec4 u_color;
out vec4 f_color;
void main() {
    vec4 t = texture(u_tex, v_uv);
    f_color = t * u_color;
}
"""

# RGBA ZERO-COPY PIPELINE (2026-08-28): a 1:1 NEAREST V-flip pass copies the
# scene FBO into a second FBO so the glReadPixels byte stream comes out
# TOP-DOWN already — the CPU never pays the flip (the old path returned a
# negative-stride view whose flip cost resurfaced as a 6 MB copy somewhere
# downstream every frame). Pixel-exact: 1:1 texel mapping, NEAREST, blending
# disabled during the pass.
_FLIP_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

_FLIP_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D u_tex;
out vec4 f_color;
void main() {
    f_color = texture(u_tex, v_uv);
}
"""


# ---- INSTANCED SPRITE PATH (R3D_INSTANCED_DRAW=1) -----------------------------
# The per-sprite path issues one draw call plus 4-6 Python uniform writes per
# sprite; at a measured 406 sprites/frame that is ~2436 Python->C calls and
# ~1.8 ms/frame. This path uploads one interleaved instance buffer and issues
# ONE draw per blend group.
#
# Textures: the engine uses ~11 distinct textures per frame and GL 3.3
# guarantees 16 image units, so every texture in a batch binds to its own unit
# and a per-instance index selects it. Sampling stays BIT-IDENTICAL to the
# per-sprite path (same texture objects, same filters) -- unlike an atlas, which
# bleeds between neighbours at coarse mip levels. A batch needing a 17th texture
# is flushed first, so sprites are never reordered.
_I_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
in vec2 i_center;
in vec2 i_size;
in float i_rot;
in vec4 i_color;
in vec2 i_uv_off;
in vec2 i_uv_scale;
in float i_tex;
uniform vec2 u_screen;
out vec2 v_uv;
out vec4 v_color;
flat out int v_tex;
void main() {
    vec2 p = in_pos * i_size;
    float c = cos(i_rot), s = sin(i_rot);
    p = vec2(p.x * c - p.y * s, p.x * s + p.y * c);
    vec2 px = i_center + p;
    vec2 ndc = vec2(px.x / u_screen.x * 2.0 - 1.0,
                    1.0 - px.y / u_screen.y * 2.0);
    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv * i_uv_scale + i_uv_off;
    v_color = i_color;
    v_tex = int(i_tex);
}
"""

_I_FRAG = """
#version 330
in vec2 v_uv;
in vec4 v_color;
flat in int v_tex;
out vec4 f_color;
uniform sampler2D u_tex0;
uniform sampler2D u_tex1;
uniform sampler2D u_tex2;
uniform sampler2D u_tex3;
uniform sampler2D u_tex4;
uniform sampler2D u_tex5;
uniform sampler2D u_tex6;
uniform sampler2D u_tex7;
uniform sampler2D u_tex8;
uniform sampler2D u_tex9;
uniform sampler2D u_tex10;
uniform sampler2D u_tex11;
uniform sampler2D u_tex12;
uniform sampler2D u_tex13;
uniform sampler2D u_tex14;
uniform sampler2D u_tex15;
void main() {
    int t = v_tex;
    vec4 tc = vec4(0.0);
    if (t == 0) tc = texture(u_tex0, v_uv);
    else if (t == 1) tc = texture(u_tex1, v_uv);
    else if (t == 2) tc = texture(u_tex2, v_uv);
    else if (t == 3) tc = texture(u_tex3, v_uv);
    else if (t == 4) tc = texture(u_tex4, v_uv);
    else if (t == 5) tc = texture(u_tex5, v_uv);
    else if (t == 6) tc = texture(u_tex6, v_uv);
    else if (t == 7) tc = texture(u_tex7, v_uv);
    else if (t == 8) tc = texture(u_tex8, v_uv);
    else if (t == 9) tc = texture(u_tex9, v_uv);
    else if (t == 10) tc = texture(u_tex10, v_uv);
    else if (t == 11) tc = texture(u_tex11, v_uv);
    else if (t == 12) tc = texture(u_tex12, v_uv);
    else if (t == 13) tc = texture(u_tex13, v_uv);
    else if (t == 14) tc = texture(u_tex14, v_uv);
    else if (t == 15) tc = texture(u_tex15, v_uv);
    f_color = tc * v_color;
}
"""

_I_STRIDE = 14          # center2 size2 rot1 color4 uv_off2 uv_scale2 tex1
_I_MAX_TEX = 16


import time as _rp_time
_READ_PERF = os.environ.get("R3D_READ_PERF") == "1"
# MAPPED READBACK (R3D_MAP_READBACK=1, darwin only for now).
# glGetBufferSubData is specified to synchronise and measured 0.663 ms standalone
# / 1.243 ms in-engine for 8.29 MB. glMapBufferRange(READ|WRITE) measured 0.316 ms
# AND removes the host copy entirely -- the mapped pointer is wrapped as a numpy
# view and handed straight to the HUD and the writer (verified: bytes through the
# map are identical to glGetBufferSubData).
# moderngl exposes no mapping API, so we call GL directly via ctypes.
_MAP_READBACK = (os.environ.get("R3D_MAP_READBACK") == "1"
                 and sys.platform == "darwin")
_GL_PIXEL_PACK_BUFFER = 0x88EB
_GL_MAP_READ_BIT      = 0x0001
_GL_MAP_WRITE_BIT     = 0x0002
_gl_c = None


def _load_gl_c():
    """ctypes handle on the already-loaded GL, for the two calls moderngl lacks."""
    global _gl_c
    if _gl_c is None:
        import ctypes
        lib = ctypes.CDLL("/System/Library/Frameworks/OpenGL.framework/OpenGL")
        lib.glBindBuffer.argtypes = [ctypes.c_uint, ctypes.c_uint]
        lib.glBindBuffer.restype = None
        lib.glMapBufferRange.argtypes = [ctypes.c_uint, ctypes.c_ssize_t,
                                         ctypes.c_ssize_t, ctypes.c_uint]
        lib.glMapBufferRange.restype = ctypes.c_void_p
        lib.glUnmapBuffer.argtypes = [ctypes.c_uint]
        lib.glUnmapBuffer.restype = ctypes.c_ubyte
        _gl_c = lib
    return _gl_c
_DRAW_ABL = os.environ.get("R3D_DRAW_ABL", "").strip()   # ablation probe only
_I_ORPHAN = os.environ.get("R3D_I_ORPHAN", "0") == "1"
try:
    _READ_SPLIT = max(1, int(os.environ.get("R3D_READ_SPLIT", "1") or 1))
except ValueError:
    _READ_SPLIT = 1
try:
    _READ_FRAC = float(os.environ.get("R3D_READ_FRAC", "1") or 1)
except ValueError:
    _READ_FRAC = 1.0
_rp_pc = _rp_time.perf_counter
_rp_perf = {"flip": 0.0, "readpix": 0.0, "pbo2host": 0.0, "n": 0}


class SpriteRenderer:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # Honor R3D_EGL_DEVICE_INDEX so renders pin to the right GPU (pool
        # isolation: e.g. 1070=index 1 for Pool B). EGL ignores
        # CUDA_VISIBLE_DEVICES, so the device must be selected explicitly.
        import os, sys
        if sys.platform in ("win32", "darwin"):
            # darwin: glcontext has no EGL backend on macOS either; the default
            # standalone context is CGL (OpenGL 4.1 core, Apple Metal-backed).
            # Windows contributors: glcontext ships no EGL backend (ImportError
            # cannot import name egl), so use the default WGL standalone context.
            # R3D_EGL_DEVICE_INDEX is an EGL-only GPU pin (Linux pools).
            self.ctx = moderngl.create_context(standalone=True)
        else:
            dev = os.environ.get("R3D_EGL_DEVICE_INDEX", "").strip()
            if dev.isdigit():
                self.ctx = moderngl.create_context(
                    standalone=True, backend="egl", device_index=int(dev))
            else:
                self.ctx = moderngl.create_context(standalone=True, backend="egl")
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.prog = self.ctx.program(vertex_shader=_VERT, fragment_shader=_FRAG)
        # unit quad centered at origin, uv 0..1
        # in_pos.y=-0.5 renders at screen-top -> texture-top (v=0); in_pos.y=+0.5
        # renders at screen-bottom -> texture-bottom (v=1). (Matters for
        # vertically-asymmetric sprites like the catcher.)
        quad = np.array([
            -0.5, -0.5, 0.0, 0.0,
             0.5, -0.5, 1.0, 0.0,
            -0.5,  0.5, 0.0, 1.0,
             0.5,  0.5, 1.0, 1.0,
        ], dtype="f4")
        self.vbo = self.ctx.buffer(quad.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog, [(self.vbo, "2f 2f", "in_pos", "in_uv")],
        )
        self.prog["u_screen"].value = (float(width), float(height))
        # PERF: resolve uniform handles once (prog[...] is a dict lookup +
        # wrapper build per call) and set the sampler unit a single time —
        # it is always texture unit 0. GL state/output is unchanged.
        self.prog["u_tex"].value = 0
        self._u_color = self.prog["u_color"]
        self._u_center = self.prog["u_center"]
        self._u_size = self.prog["u_size"]
        self._u_rot = self.prog["u_rot"]
        # UV flip uniforms (storyboard mirroring). Bound to identity so every
        # gameplay/HUD sprite (uv_off=(0,0), uv_scale=(1,1)) samples exactly
        # `in_uv` — bit-identical to before. _draw_one only re-binds them when a
        # sprite's uv differs from what's currently bound, so a flag-off render
        # never touches them after this init (zero per-sprite overhead too).
        self._u_uv_off = self.prog["u_uv_off"]
        self._u_uv_scale = self.prog["u_uv_scale"]
        self._u_uv_off.value = (0.0, 0.0)
        self._u_uv_scale.value = (1.0, 1.0)
        self._cur_uv_off = (0.0, 0.0)
        self._cur_uv_scale = (1.0, 1.0)

        # Scene target is a TEXTURE (not a renderbuffer) so the flip pass can
        # sample it. Rendering into a texture attachment is identical to a
        # renderbuffer attachment. NEAREST filter: the flip pass must be a
        # pixel-exact 1:1 copy.
        self._scene_tex = self.ctx.texture((width, height), 4)
        self._scene_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.fbo = self.ctx.framebuffer(color_attachments=[self._scene_tex])
        # flip-pass target: readback happens from here (top-down byte order).
        self._flip_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer((width, height))])
        self._flip_prog = self.ctx.program(vertex_shader=_FLIP_VERT,
                                           fragment_shader=_FLIP_FRAG)
        self._flip_prog["u_tex"].value = 0
        # Fullscreen strip mapping output row 0 (GL bottom, read FIRST by
        # glReadPixels) to the scene's TOP row (v=1): the packed stream is
        # top-left origin without any CPU flip.
        _fq = np.array([
            -1.0, -1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0,
            -1.0,  1.0, 0.0, 0.0,
             1.0,  1.0, 1.0, 0.0,
        ], dtype="f4")
        self._flip_vbo = self.ctx.buffer(_fq.tobytes())
        self._flip_vao = self.ctx.vertex_array(
            self._flip_prog, [(self._flip_vbo, "2f 2f", "in_pos", "in_uv")],
        )
        self._textures: dict[str, moderngl.Texture] = {}
        self._white = self._make_texture_rgba(np.full((1, 1, 4), 255, dtype="u1"))

        # async PBO ring state (read_rgb_async / read_drain) — ported from
        # the std renderer's proven pipeline (osu_std_renderer/render/gl.py)
        self._pbos: list["moderngl.Buffer"] | None = None
        self._pbo_head = 0
        self._pbo_tail = 0

    # --- texture management ---------------------------------------------------

    def upload_texture(self, key: str, rgba: np.ndarray,
                       clamp: bool = False, mipmaps: bool = True) -> None:
        """rgba: HxWx4 uint8 array (top-left origin). clamp=True sets
        clamp-to-edge wrapping (the storyboard samples with a flipped UV that
        can graze the edge texel; repeat wrap would wrap the far edge in).
        mipmaps default True — existing callers pass neither and are unchanged
        (mipmapped LINEAR, repeat wrap, exactly as before)."""
        if rgba.dtype != np.uint8:
            rgba = rgba.astype("u1")
        if rgba.shape[2] == 3:
            a = np.full(rgba.shape[:2] + (1,), 255, dtype="u1")
            rgba = np.concatenate([rgba, a], axis=2)
        tex = self._make_texture_rgba(rgba, mipmaps=mipmaps)
        if clamp:
            tex.repeat_x = False
            tex.repeat_y = False
        self._textures[key] = tex

    def has_texture(self, key: str) -> bool:
        return key in self._textures

    def release_texture(self, key: str) -> None:
        """Free a cached texture by key (storyboard LRU eviction). No-op if
        the key is absent."""
        tex = self._textures.pop(key, None)
        if tex is not None:
            try:
                tex.release()
            except Exception:  # noqa: BLE001 - context may be tearing down
                pass

    def _make_texture_rgba(self, rgba: np.ndarray,
                           mipmaps: bool = True) -> "moderngl.Texture":
        h, w = rgba.shape[:2]
        tex = self.ctx.texture((w, h), 4, rgba.tobytes())
        if mipmaps:
            tex.build_mipmaps()
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        else:
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        return tex

    # --- drawing --------------------------------------------------------------

    def begin(self, clear=(0.04, 0.04, 0.06)) -> None:
        self.fbo.use()
        self.ctx.clear(*clear)

    def draw(self, sprites: list[Sprite]) -> None:
        if _DRAW_COUNT:
            _DRAW_STATS[0] += 1
            _DRAW_STATS[1] += len(sprites)
            if len(sprites) > _DRAW_STATS[2]:
                _DRAW_STATS[2] = len(sprites)
            # instancing can only batch sprites that SHARE a texture, so the
            # number of distinct texture keys per frame bounds the draw-call
            # count after batching
            _keys = {sp.texture_key for sp in sprites}
            _DRAW_STATS[3] += len(_keys)
            if len(_keys) > _DRAW_STATS[4]:
                _DRAW_STATS[4] = len(_keys)
            _DRAW_STATS[5] += sum(1 for sp in sprites if sp.additive)
            # ORDER-PRESERVING instancing can only batch CONSECUTIVE runs of
            # the same texture, because painter's order decides alpha layering.
            # This counts those runs = the real draw-call count after batching.
            _runs = 0
            _prev = object()
            for sp in sprites:
                _k = (sp.texture_key, sp.additive)
                if _k != _prev:
                    _runs += 1
                    _prev = _k
            _DRAW_STATS[6] += _runs
            if _runs > _DRAW_STATS[7]:
                _DRAW_STATS[7] = _runs
        # Painter's order for the normal (straight-alpha) sprites, then an
        # additive pass on top for glow/explosion sprites (lazer draws hit
        # explosions & catcher trails additively).
        add = []
        if _INSTANCED:
            add = [sp for sp in sprites if sp.additive]
            self._draw_instanced([sp for sp in sprites if not sp.additive])
        else:
            for sp in sprites:
                if sp.additive:
                    add.append(sp)
                else:
                    self._draw_one(sp)
        if _DRAW_FINISH and not add:
            self.ctx.finish()
        if add:
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
            if _INSTANCED:
                self._draw_instanced(add)
            else:
                for sp in add:
                    self._draw_one(sp)
            self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

    def _ensure_instanced(self):
        if getattr(self, "_i_prog", None) is not None:
            return
        self._i_prog = self.ctx.program(vertex_shader=_I_VERT,
                                        fragment_shader=_I_FRAG)
        self._i_prog["u_screen"].value = (float(self.width), float(self.height))
        for u in range(_I_MAX_TEX):
            self._i_prog[f"u_tex{u}"].value = u
        self._i_cap = 2048
        self._i_alloc()

    def _i_alloc(self):
        self._i_buf = self.ctx.buffer(reserve=self._i_cap * _I_STRIDE * 4)
        self._i_vao = self.ctx.vertex_array(self._i_prog, [
            (self.vbo, "2f 2f", "in_pos", "in_uv"),
            (self._i_buf, "2f 2f 1f 4f 2f 2f 1f/i", "i_center", "i_size",
             "i_rot", "i_color", "i_uv_off", "i_uv_scale", "i_tex"),
        ])
        self._i_scratch = np.empty((self._i_cap, _I_STRIDE), dtype="f4")

    def _draw_instanced(self, sprites) -> None:
        """One draw per blend group; order preserved within the group."""
        n = len(sprites)
        if n == 0:
            return
        self._ensure_instanced()
        if n > self._i_cap:
            self._i_cap = 1 << (n - 1).bit_length()
            self._i_alloc()
        arr = self._i_scratch
        state = {"units": {}}
        # PERF (bit-identical): the old fill did 14 numpy scalar setitems per
        # sprite -- ~5.7k element assignments/frame at ~120 ns each, which WAS
        # the whole `draw` cost. Accumulate into a flat Python list (one
        # list.extend per sprite) and convert once; numpy's list->float32
        # conversion rounds each value exactly as the per-element store did.
        vals: list = []
        flat = arr.reshape(-1)

        def flush():
            c = len(vals) // _I_STRIDE
            if not c:
                return
            if _DRAW_ABL != "all":
                flat[:c * _I_STRIDE] = vals
                if _DRAW_ABL not in ("write", "texuse"):
                    # PERF (bit-identical): `arr[:c]` is a C-contiguous view, so
                    # moderngl can consume it via the buffer protocol directly.
                    # .tobytes() allocated + copied ~23 KB per frame and measured
                    # 0.255 ms of the 0.668 ms draw -- call overhead, not bandwidth.
                    if _I_ORPHAN:
                        # glBufferSubData STALLS while the GPU is still reading
                        # last frame's instance data. orphan() discards the old
                        # storage so the driver hands back fresh memory instead
                        # of waiting. Standard streaming-buffer trick; contents
                        # written are unchanged, so output is unaffected.
                        self._i_buf.orphan()
                    self._i_buf.write(arr[:c])
                if _DRAW_ABL == "":
                    self._i_vao.render(moderngl.TRIANGLE_STRIP, instances=c)
            del vals[:]
            state["units"] = {}

        extend = vals.extend
        for sp in sprites:
            tex = self._textures.get(sp.texture_key) if sp.texture_key \
                else self._white
            if tex is None:
                tex = self._white
            units = state["units"]
            u = units.get(id(tex))
            if u is None:
                if len(units) >= _I_MAX_TEX:      # 17th texture: flush, keep order
                    flush()
                    units = state["units"]
                u = len(units)
                units[id(tex)] = u
                if _DRAW_ABL not in ("texuse", "all"):
                    tex.use(location=u)
            c0, c1, c2, c3 = sp.color
            o0, o1 = sp.uv_off
            s0, s1 = sp.uv_scale
            extend((sp.x, sp.y, sp.w, sp.h, sp.rotation,
                    c0, c1, c2, c3, o0, o1, s0, s1, u))
        flush()

    def _draw_one(self, sp: Sprite) -> None:
        tex = self._textures.get(sp.texture_key) if sp.texture_key else self._white
        if tex is None:
            tex = self._white
        tex.use(location=0)
        self._u_color.value = sp.color
        self._u_center.value = (sp.x, sp.y)
        self._u_size.value = (sp.w, sp.h)
        self._u_rot.value = sp.rotation
        # UV flip (storyboard mirroring). Every gameplay/HUD sprite keeps the
        # identity default, so these never re-bind on a flag-off render and the
        # sampled UV stays exactly `in_uv` — byte-identical to before.
        if sp.uv_off != self._cur_uv_off:
            self._u_uv_off.value = sp.uv_off
            self._cur_uv_off = sp.uv_off
        if sp.uv_scale != self._cur_uv_scale:
            self._u_uv_scale.value = sp.uv_scale
            self._cur_uv_scale = sp.uv_scale
        if not _DRAW_NORENDER:
            self.vao.render(moderngl.TRIANGLE_STRIP)

    # Ring depth: how many frames of slack the async glReadPixels gets before
    # read_into must wait for it. Tested FLAT standalone (2-12, 9% spread) but
    # standalone the fixed cost is only ~0.38 ms; in-engine it is ~0.98 ms, so
    # the sync theory is only testable here. R3D_PBO_RING overrides.
    # CLAMPED: _HOST_POOL (24) is budgeted for ring=3 -- ring + composite
    # queue(3) + in-process(1) + writer queue(12) + in-write(1) + slack. A ring
    # above ~6 overruns it, so a queued frame's host buffer is overwritten
    # before the writer consumes it and the OUTPUT SILENTLY CORRUPTS (measured:
    # ring=12 changed the mp4 md5; ring=16 did not, i.e. non-deterministic).
    # Deeper rings are also slower in-engine (pbo2host 1.269 ms at 2 -> 1.800 at
    # 16), so there is no reason to want one.
    try:
        _PBO_RING = min(6, max(2, int(os.environ.get("R3D_PBO_RING", "3") or 3)))
    except ValueError:
        _PBO_RING = 3
    # RGBA zero-copy: the frames handed out by _pop_pbo are now mutated in
    # place by the HUD and pushed AS-IS through the composite queue (3) AND
    # the writer queue (12) — the pool must outlive every queued reference:
    # ring(3) + composite queue(3) + composite in-process(1) + writer
    # queue(12) + writer in-write(1) + slack. 24 x ~8.3 MB @1080p ≈ 200 MB.
    _HOST_POOL = 24

    def gl_hud_stub(self, n: int) -> None:
        """PROBE (R3D_GL_STUB=n): issue n extra full-screen draws on the render
        thread to price what a GL HUD would COST there, without writing pixels.
        Scissored to a 1x1 corner so output is unchanged."""
        if n <= 0:
            return
        self.ctx.scissor = (0, 0, 1, 1)
        try:
            for _ in range(n):
                self._flip_vao.render(moderngl.TRIANGLE_STRIP)
        finally:
            self.ctx.scissor = None

    def read_rgb_async(self) -> "np.ndarray | None":
        """Queue an async readback of the current frame into a small PBO
        ring and return the OLDEST completed frame, or None while the ring
        is still filling. Frames come back in strict submission order —
        the render loop pushes them straight to the compositor. RGBA
        ZERO-COPY PIPELINE: a GPU flip pass makes the packed stream
        top-left origin, so the returned array is a top-down CONTIGUOUS
        WRITABLE HxWx4 view of a pooled staging buffer — the HUD wraps it
        in PIL zero-copy, draws in place, and the SAME buffer flows to the
        ffmpeg pipe (-pix_fmt rgba). read_drain() flushes the tail."""
        if self._pbos is None:
            size = self.width * self.height * 4
            self._pbo_size = size
            # Mapped mode: the PBOs ARE the host buffers, so we need one per
            # frame in flight (ring 3 + composite queue 3 + in-process 1 +
            # writer queue 12 + in-write 1 + slack) -- the same budget the host
            # pool used. `_HOST_POOL` then disappears, which retires the
            # ring>pool corruption hazard by construction.
            _n = self._HOST_POOL if _MAP_READBACK else self._PBO_RING
            self._pbos = [self.ctx.buffer(reserve=size) for _ in range(_n)]
            self._mapped = [False] * _n
            # readback LATENCY stays 3 regardless of how many buffers exist
            self._lat = self._PBO_RING
            self._pbo_host = (None if _MAP_READBACK
                              else [bytearray(size) for _ in range(self._HOST_POOL)])
            self._host_i = 0
        if _READ_PERF:
            _p = _rp_perf; _c = _rp_pc
            _a = _c()
            self._flip_fbo.use()
            self.ctx.disable(moderngl.BLEND)
            self._scene_tex.use(location=0)
            self._flip_vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.enable(moderngl.BLEND)
            _b = _c(); _p["flip"] += _b - _a
            buf = self._unmap_for_write(self._pbo_head % len(self._pbos))
            self._flip_fbo.read_into(buf, components=4, alignment=1)
            _d = _c(); _p["readpix"] += _d - _b
            self._pbo_head += 1
            if self._pbo_head - self._pbo_tail < self._lat:
                return None
            r = self._pop_pbo()
            _p["pbo2host"] += _c() - _d
            _p["n"] += 1
            return r
        # GPU flip pass (blending OFF: raw texel copy, pixel-exact 1:1).
        self._flip_fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self._scene_tex.use(location=0)
        self._flip_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.enable(moderngl.BLEND)
        buf = self._unmap_for_write(self._pbo_head % len(self._pbos))
        self._flip_fbo.read_into(buf, components=4, alignment=1)
        self._pbo_head += 1
        if self._pbo_head - self._pbo_tail < self._lat:
            return None
        return self._pop_pbo()

    def _unmap_for_write(self, idx):
        """A mapped buffer cannot receive glReadPixels. Unmap on REUSE -- by the
        time the ring wraps back to this index the writer is long done with it,
        so no completion callback is needed."""
        buf = self._pbos[idx]
        if _MAP_READBACK and self._mapped[idx]:
            g = _load_gl_c()
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, buf.glo)
            g.glUnmapBuffer(_GL_PIXEL_PACK_BUFFER)
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, 0)
            self._mapped[idx] = False
        return buf

    def _pop_pbo(self) -> np.ndarray:
        if _MAP_READBACK:
            import ctypes as _ct
            idx = self._pbo_tail % len(self._pbos)
            buf = self._pbos[idx]
            self._pbo_tail += 1
            g = _load_gl_c()
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, buf.glo)
            ptr = g.glMapBufferRange(_GL_PIXEL_PACK_BUFFER, 0, self._pbo_size,
                                     _GL_MAP_READ_BIT | _GL_MAP_WRITE_BIT)
            g.glBindBuffer(_GL_PIXEL_PACK_BUFFER, 0)
            if not ptr:
                raise RuntimeError("glMapBufferRange returned NULL")
            self._mapped[idx] = True
            arr = np.ctypeslib.as_array(
                (_ct.c_uint8 * self._pbo_size).from_address(ptr))
            return arr.reshape((self.height, self.width, 4))
        buf = self._pbos[self._pbo_tail % len(self._pbos)]
        self._pbo_tail += 1
        host = self._pbo_host[self._host_i % len(self._pbo_host)]
        self._host_i += 1
        if _READ_SPLIT > 1:
            # PROBE: same bytes, N calls. If the fixed cost is PER-CALL it
            # multiplies; if it is a one-time sync it stays flat. Output is
            # correct either way -- the halves are contiguous and complete.
            n = len(host); step = n // _READ_SPLIT // 4 * 4
            mv = memoryview(host)
            off = 0
            for _k in range(_READ_SPLIT):
                sz = step if _k < _READ_SPLIT - 1 else n - off
                buf.read_into(mv[off:off + sz], size=sz, offset=off)
                off += sz
        elif _READ_FRAC < 1.0:
            # TIMING PROBE ONLY (R3D_READ_FRAC): copy a FRACTION of the bytes to
            # find out whether pbo2host is bytes-bound or fixed-cost-bound under
            # real in-engine contention. Output is deliberately WRONG below 1.0.
            buf.read_into(host, size=int(len(host) * _READ_FRAC) // 4 * 4)
        else:
            buf.read_into(host)
        # top-left origin already (GPU flip pass) — contiguous + writable,
        # ready for the zero-copy PIL wrap. NOTE the alpha channel carries
        # GL blend garbage; ffmpeg's rgba input ignores it, and every CPU
        # consumer that READS the canvas forces alpha itself.
        return np.frombuffer(host, dtype="u1").reshape(
            (self.height, self.width, 4))

    def read_drain(self) -> list:
        """Return every frame still in flight, oldest first (map end or
        the gameplay->outro boundary)."""
        out = []
        while self._pbos is not None and self._pbo_tail < self._pbo_head:
            out.append(self._pop_pbo())
        return out

    def read_rgb(self) -> np.ndarray:
        """Return HxWx3 uint8, top-left origin (ready for ffmpeg rgb24)."""
        data = self.fbo.read(components=3, alignment=1)
        arr = np.frombuffer(data, dtype="u1").reshape((self.height, self.width, 3))
        # moderngl reads bottom-left origin; flip to top-left
        return np.flipud(arr)

    def release(self) -> None:
        try:
            self.ctx.release()
        except Exception:  # noqa: BLE001
            pass


def read_perf_report():
    """Per-phase readback breakdown (R3D_READ_PERF=1)."""
    p = _rp_perf; n = max(1, p["n"])
    return ("READ-PERF n=%d flip=%.3f readpix=%.3f pbo2host=%.3f total=%.3f ms/frame"
            % (p["n"], p["flip"]/n*1e3, p["readpix"]/n*1e3,
               p["pbo2host"]/n*1e3,
               (p["flip"]+p["readpix"]+p["pbo2host"])/n*1e3))
