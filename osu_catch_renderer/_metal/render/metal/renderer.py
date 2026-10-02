"""MetalSpriteRenderer — drop-in for gl.SpriteRenderer on Apple Silicon.

Why: the GL path's per-frame GPU->CPU readback (glGetBufferSubData) measured
1.38-1.56 ms in-engine and resisted every knob (byte count, ring depth, call
splitting, alignment, dtype, pool size). Metal renders straight into a
buffer-backed MTLTexture over .storageModeShared, so the CPU reads the pixels
through a pointer with NO copy. Measured, 406 sprites + readback:

    GL in-engine (draw 0.623 + read 1.874)   2.497 ms
    Metal idle                               0.127 ms   19.6x
    Metal under 86% CPU load                 0.230 ms   10.9x

Output is NOT byte-identical to GL (different rasteriser, and no mipmaps yet --
see upload_texture). Cross-vendor byte-identity is already broken anyway
(Apple 0/3733 vs NVIDIA, AMD 3487/3733 vs NVIDIA), so this is judged on a
fidelity band, not equality.
"""
from __future__ import annotations

import os
import numpy as np

from osu_catch_renderer._metal.render.metal.metal_gl import MetalRenderer

# Max texture units per draw, matching gl.py's _I_MAX_TEX. A batch needing a
# 17th texture flushes first, so painter's order is never reordered.
_MAX_TEX = 16
_I_MAX_TEX = 16      # texture units per draw, as in gl.py
_STRIDE = 14


class MetalSpriteRenderer:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        # Ring must outlive every queued reference downstream: composite
        # queue(3) + in-process(1) + writer queue(12) + in-write(1) + slack.
        # Same budget as gl._HOST_POOL; undersizing it corrupts frames.
        ring = int(os.environ.get("R3D_METAL_RING", "24") or 24)
        self.r = MetalRenderer(width, height, ring=ring,
                               copy_out=os.environ.get("R3D_METAL_ZEROCOPY") != "1")
        self.device = self.r.device
        self._tex: dict[str, int] = {}
        self._white = self.r.tex_create(np.full((2, 2, 4), 255, np.uint8))
        self._scratch = np.empty((4096, _STRIDE), np.float32)

    # ---- texture management (same signatures as gl.SpriteRenderer) ----------
    def upload_texture(self, key: str, rgba: np.ndarray,
                       clamp: bool = False, mipmaps: bool = True) -> None:
        # `mipmaps` is honoured (blit-encoder generateMipmaps at upload time,
        # sampler mipFilter=.linear) because GL uses mipmapped LINEAR and the
        # background is a heavily-downscaled full-screen image -- without mips
        # the whole frame aliased (PSNR 19-28 dB vs GL). `clamp` is moot: the
        # sampler already clamps to edge.
        if rgba.dtype != np.uint8:
            rgba = rgba.astype("u1")
        if rgba.shape[2] == 3:
            a = np.full(rgba.shape[:2] + (1,), 255, dtype="u1")
            rgba = np.concatenate([rgba, a], axis=2)
        self._tex[key] = self.r.tex_create(rgba, mipmaps=mipmaps)

    def has_texture(self, key: str) -> bool:
        return key in self._tex

    def release_texture(self, key: str) -> None:
        self._tex.pop(key, None)      # dylib frees on destroy

    # ---- frame lifecycle ----------------------------------------------------
    def begin(self, clear=(0.04, 0.04, 0.06)) -> None:
        self.r.begin((clear[0], clear[1], clear[2], 1.0))

    def draw(self, sprites) -> None:
        """Two passes (normal then additive), preserving painter's order inside
        each -- identical batching rules to gl._draw_instanced."""
        add = [sp for sp in sprites if sp.additive]
        norm = [sp for sp in sprites if not sp.additive]
        if norm:
            self._draw_group(norm, additive=False)
        if add:
            self._draw_group(add, additive=True)

    def _draw_group(self, sprites, additive: bool) -> None:
        n = len(sprites)
        if n > self._scratch.shape[0]:
            self._scratch = np.empty((1 << (n - 1).bit_length(), _STRIDE),
                                     np.float32)
        arr = self._scratch
        vals: list = []
        units: dict[int, int] = {}
        ids: list[int] = []

        def flush():
            c = len(vals) // _STRIDE
            if not c:
                return
            arr.reshape(-1)[:c * _STRIDE] = vals
            self.r.draw(arr[:c], ids, additive=additive)
            del vals[:]
            units.clear()
            del ids[:]

        extend = vals.extend
        for sp in sprites:
            tid = self._tex.get(sp.texture_key, self._white) \
                if sp.texture_key else self._white
            u = units.get(tid)
            if u is None:
                if len(units) >= _MAX_TEX:      # 17th texture: flush, keep order
                    flush()
                u = len(units)
                units[tid] = u
                ids.append(tid)
            c0, c1, c2, c3 = sp.color
            o0, o1 = sp.uv_off
            s0, s1 = sp.uv_scale
            extend((sp.x, sp.y, sp.w, sp.h, sp.rotation,
                    c0, c1, c2, c3, o0, o1, s0, s1, u))
        flush()

    # ---- in-pass GPU HUD ---------------------------------------------------
    def hud_gpu_init(self, argon_hp, argon_hud):
        """Prepare the in-pass HUD: upload the static SDF planes and the wedge
        tile, and precompute the wedge rects. Everything here is static for the
        whole render, so the per-frame cost is 2 sprites + 3 blended draws."""
        import numpy as _np
        from osu_catch_renderer._metal.argon.argon_hud import WEDGE_H, WEDGE_SHEAR
        M, G = argon_hp.main, argon_hp.glow
        self.r.hp_init([M.d_perp, M.t_near, G.d_perp, G.t_near])
        w_img = argon_hud._wedge
        self._wedge_tex = self.r.tex_create(
            _np.ascontiguousarray(_np.asarray(w_img.convert("RGBA"))),
            mipmaps=False)
        k = argon_hud.es * argon_hud.lk
        tw, th = w_img.size
        self._wedge_rects = []
        for px, py in ((-50.0, 15.0), (-46.0, 20.0)):
            x = int(round((px - WEDGE_SHEAR * WEDGE_H) * k))
            y = int(round(py * k))
            # sprite geometry is CENTRE-based; paste is top-left
            self._wedge_rects.append((x + tw * 0.5, y + th * 0.5, tw, th))
        self._hp_ready = True

    def draw_hud_gpu(self, params):
        """Wedges then the bar, in the open pass. Order matches hud.overlay:
        wedges are UNDER the bar (they overlap by ~464x101 px)."""
        import numpy as _np
        if getattr(self, "_wedge_rects", None):
            a = _np.zeros((len(self._wedge_rects), _STRIDE), _np.float32)
            for i, (cx, cy, w, h) in enumerate(self._wedge_rects):
                a[i, 0:4] = (cx, cy, w, h)
                a[i, 5:9] = 1.0
                a[i, 11:13] = 1.0
            self.r.draw(a, [self._wedge_tex], additive=False)
        self.r.hp_inline(params)

    def progress_gpu_init(self, ah):
        """Upload the two STATIC progress layers. The density graph and the bg
        pill never change for a render; only the fill pill's width moves, and
        that is a shader (an exact port of bake_pill_alpha)."""
        import numpy as _np
        from osu_catch_renderer._metal.argon.argon_hud import bake_pill_alpha, _AA_PX
        sx0, sx1, sy0, sy1 = ah._sx0, ah._sx1, ah._sy0, ah._sy1
        sw, sh = max(1, sx1-sx0), max(1, sy1-sy0)
        self._pg_rect = (sx0 + sw*0.5, sy0 + sh*0.5, sw, sh)
        self._pg_geom = (sx0, sy0, sw, sh)
        self._pg_aa = _AA_PX
        self._pg_op = ah.op
        # density graph -> additive sprite: out = dst + rgb*255, so rgb = add/255
        g = _np.asarray(ah._graph_add, dtype=_np.float64)
        g = _np.broadcast_to(g, (sh, sw, 3)) if g.shape[-1] == 1 else g
        gt = _np.empty((sh, sw, 4), _np.uint8)
        gt[..., :3] = _np.clip(_np.rint(g), 0, 255).astype(_np.uint8)
        gt[..., 3] = 255
        self._pg_graph_tex = self.r.tex_create(gt, mipmaps=False)
        # bg pill -> src-over sprite: rgb 0.2, a = pill*0.3*op
        a = bake_pill_alpha(sw, sh) * (0.3 * ah.op)
        bt = _np.empty((sh, sw, 4), _np.uint8)
        bt[..., 0] = bt[..., 1] = bt[..., 2] = int(round(0.2 * 255.0))
        bt[..., 3] = _np.clip(_np.rint(a * 255.0), 0, 255).astype(_np.uint8)
        self._pg_bg_tex = self.r.tex_create(bt, mipmaps=False)
        self._pg_ready = True
    def draw_progress_gpu(self, frac):
        """graph (additive) -> bg pill (src-over) -> fill pill (src-over)."""
        import numpy as _np
        cx, cy, sw, sh = self._pg_rect
        sx0, sy0, _, _ = self._pg_geom
        inst = _np.zeros((1, _STRIDE), _np.float32)
        inst[0, 0:4] = (cx, cy, sw, sh)
        inst[0, 5:9] = 1.0
        inst[0, 11:13] = 1.0
        self.r.draw(inst, [self._pg_graph_tex], additive=True)
        self.r.draw(inst, [self._pg_bg_tex], additive=False)
        if frac > 0.003:
            fw = max(1, int(round(sw * frac)))
            self.r.pill_inline([self.width, self.height, sx0, sy0, fw, sh,
                                self._pg_aa, 0.95 * self._pg_op, 0.9, 0.9, 0.9])

    # ---- GPU results screen ------------------------------------------------
    def results_gpu_init(self):
        self._rs_tex = {}          # id(PIL image) -> texture id
        self._rs_keep = []         # hold references so id() stays valid
        self._rs_frozen = None

    def results_tile(self, img):
        """Texture for a baked results tile, uploaded on first sight. Keyed by
        id(), which is safe because every tile is owned by the results-screen
        instance or the shared arc/score prebake dicts, so none are collected."""
        import numpy as _np
        k = id(img)
        t = self._rs_tex.get(k)
        if t is None:
            a = _np.asarray(img.convert("RGBA"), dtype=_np.uint8)
            t = self.r.tex_create(_np.ascontiguousarray(a), mipmaps=False)
            self._rs_tex[k] = t
            self._rs_keep.append(img)
        return t

    def results_frozen(self, rgba):
        """Upload the frozen final gameplay frame once; it never changes."""
        import numpy as _np
        if self._rs_frozen is None:
            a = _np.ascontiguousarray(rgba[..., :4] if rgba.shape[2] == 4
                                      else _np.dstack([rgba, _np.full(
                                          rgba.shape[:2] + (1,), 255, _np.uint8)]))
            self._rs_frozen = self.r.tex_create(a, mipmaps=False)
        return self._rs_frozen

    def results_submit(self, ops, frozen_a, frozen_rgba):
        """Compose an outro frame and submit render+convert as ONE command
        buffer. Does not wait; collect with results_collect()."""
        self.draw_results_gpu(ops, frozen_a, frozen_rgba)
        self.r.commit_yuv()

    def results_collect(self, depth=3, force=False):
        return self.r.yuv_ring_acquire(depth, force)

    def results_frame_yuv(self, ops, frozen_a, frozen_rgba):
        """Compose an outro frame and hand back planar yuv420p, never bringing
        the RGBA across. Only safe because a GPU outro frame is COMPLETE in the
        render target -- no CPU stage follows it."""
        self.draw_results_gpu(ops, frozen_a, frozen_rgba)
        self.r.commit()
        return self.r.yuv_from_ring()

    def draw_results_gpu(self, ops, frozen_a, frozen_rgba):
        """Compose one outro frame from a captured draw list.

        `ops` is [(PIL image, x0, y0, alpha)] straight out of the results
        screen's own layout code, so geometry cannot drift from the CPU path.
        The black wash is the pass clear; the fade is the frozen frame drawn as
        one sprite at `frozen_a`, which is arithmetically what PIL's
        alpha_composite(opaque, black@A) computes.
        """
        import numpy as _np
        self.begin(clear=(0.0, 0.0, 0.0))
        if frozen_a > 0.0:
            t = self.results_frozen(frozen_rgba)
            inst = _np.zeros((1, _STRIDE), _np.float32)
            inst[0, 0:4] = (self.width * 0.5, self.height * 0.5,
                            self.width, self.height)
            inst[0, 5:8] = 1.0
            inst[0, 8] = frozen_a
            inst[0, 11:13] = 1.0
            self.r.draw(inst, [t], additive=False)
        # BATCHED: one draw call per run of <=16 distinct textures instead of
        # one per tile. 26 tiles meant 26 encoder state changes per frame.
        # Painter's order is preserved within a call, and the runs are emitted
        # in order, so the composite order is exactly the capture order.
        i = 0
        n = len(ops)
        # A throw between begin() and commit() leaves the Metal encoder open,
        # and the NEXT begin() then trips "Command encoder released without
        # endEncoding". The caller's fallback must not be able to do that, so
        # anything raised in here is committed away first.
        try:
            self._results_batches(ops)
        except Exception:
            self.r.commit()
            try:
                self.r.acquire(force=True)
            except Exception:
                pass
            raise
        return

    def _results_batches(self, ops):
        import numpy as _np
        i = 0
        n = len(ops)
        while i < n:
            texs, slot, insts = [], {}, []
            while i < n:
                img, x0, y0, a = ops[i]
                t = self.results_tile(img)
                if t not in slot:
                    if len(texs) >= _I_MAX_TEX:
                        break
                    slot[t] = len(texs)
                    texs.append(t)
                w, h = img.width, img.height
                row = _np.zeros(_STRIDE, _np.float32)
                row[0:4] = (x0 + w * 0.5, y0 + h * 0.5, w, h)
                row[5:8] = 1.0
                row[8] = min(a, 1.0)
                row[11:13] = 1.0
                row[13] = slot[t]
                insts.append(row)
                i += 1
            if insts:
                self.r.draw(_np.stack(insts), texs, additive=False)

    def read_rgb_async(self):
        """Commit the frame and return the OLDEST completed one as a zero-copy
        HxWx4 view, or None while the ring fills. Same contract as the GL path."""
        self.r.commit()
        return self.r.acquire()

    def read_drain(self) -> list:
        out = []
        while True:
            f = self.r.acquire(force=True)
            if f is None:
                return out
            out.append(f)

    def release(self) -> None:
        try:
            self.r.close()
        except Exception:      # noqa: BLE001 - teardown
            pass
