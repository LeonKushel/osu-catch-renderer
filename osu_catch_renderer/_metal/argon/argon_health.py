"""osu!lazer **Argon** health display, ported pixel-faithfully.

This is a direct port of lazer's `ArgonHealthDisplay` (C#) + its GLSL bar shader
(`sh_ArgonBarPath.fs` / `sh_ArgonBarPathUtils.h` / `sh_ArgonBarPathBackground.fs`
from ppy/osu-resources). The bar is a distance-field path: for every pixel we
find the distance to the visible segment of a centre-line, then map that distance
to colour with the exact glow falloff lazer uses.

Three stacked layers (matching lazer's container tree):
  * background  — faint dark->white track (always full length),  normal blend
  * glow bar    — fat #7ED7FD halo over the [health, recentMax] segment, additive
  * main bar    — white core + thin blue edge over [0, health],             additive

Plus the value animation: health/glow values are damped (50 ms half-life), and a
miss freezes the glow segment and flashes it red (255,93,93)/(253,0,0) before it
retracts — the signature Argon "drain trail".

All sizes are lazer's absolute pixels at a 1080p reference, scaled by s = H/1080.
"""
from __future__ import annotations

import os
import numpy as np

# --- lazer constants (absolute px @ 1080p) -----------------------------------
MAIN_PATH_RADIUS = 10.0
GLOW_PATH_RADIUS = 40.0
MAIN_GLOW_PORTION = 0.6
BAR_HEIGHT = 20.0
PADDING = MAIN_PATH_RADIUS * 2.0            # content height = BAR_HEIGHT + PADDING
DEFAULT_WIDTH = 300.0          # lazer ArgonSkin default-skin health bar Width
# curve (the right-end hook) — sh_ArgonBarPathUtils.h
CURVE_START_OFFSET = 70.0
CURVE_END_OFFSET = 40.0

GLOW_RGB = (0x7E / 255.0, 0xD7 / 255.0, 0xFD / 255.0)   # #7ED7FD
GLOW_A = 0.5
RED_BAR_RGB = (255 / 255.0, 93 / 255.0, 93 / 255.0)     # miss bar
RED_GLOW_RGB = (253 / 255.0, 0.0, 0.0)                  # miss glow

# glow bar's GlowPortion = (glow_r - main_r*(1-main_glow_portion)) / glow_r
GLOW_GLOW_PORTION = (GLOW_PATH_RADIUS - MAIN_PATH_RADIUS * (1.0 - MAIN_GLOW_PORTION)) / GLOW_PATH_RADIUS


def _damp(cur: float, target: float, half_life_ms: float, dt_ms: float) -> float:
    if dt_ms <= 0:
        return cur
    return target + (cur - target) * (0.5 ** (dt_ms / half_life_ms))


def _build_centerline(sx: float, sy: float, R: float, k: float):
    """Replicate getBarTexturePosition's centre-line as a dense polyline.

    Returns (cx, cy, ct) sample arrays with ct the cumulative arc-length
    fraction in [0,1]. The 10px corner fillets are omitted (negligible at bar
    scale); the dominant horizontal->slash->horizontal hook is exact.
    `k` scales lazer's absolute curve offsets to our bar size.
    """
    p1 = np.array([min(R, sx * 0.5), min(R, sy * 0.5)])
    p4 = np.array([max(sx - R, sx * 0.5), max(sy - R, sy * 0.5)])
    if abs(p4[1] - p1[1]) < 1e-6:
        pts = [p1, np.array([p4[0], p1[1]])]
    else:
        cso, ceo = CURVE_START_OFFSET * k, CURVE_END_OFFSET * k
        top_w = max(sx - R - cso, p1[0]) - p1[0]
        bot_w = p4[0] - max(sx - R - ceo, p1[0])
        if top_w < bot_w:
            top_w = bot_w = (top_w + bot_w) * 0.5
        p2 = np.array([p1[0] + top_w, p1[1]])
        p3 = np.array([p4[0] - bot_w, p4[1]])
        pts = [p1, p2, p3, p4]
    # densely resample the polyline by arc length
    seg = []
    cum = [0.0]
    for i in range(1, len(pts)):
        seg.append(np.linalg.norm(pts[i] - pts[i - 1]))
        cum.append(cum[-1] + seg[-1])
    total = cum[-1] or 1.0
    N = 800
    cx = np.empty(N); cy = np.empty(N); ct = np.linspace(0.0, 1.0, N)
    for k, t in enumerate(ct):
        d = t * total
        j = 1
        while j < len(cum) and cum[j] < d:
            j += 1
        j = min(j, len(pts) - 1)
        seglen = seg[j - 1] or 1.0
        f = (d - cum[j - 1]) / seglen
        p = pts[j - 1] + (pts[j] - pts[j - 1]) * f
        cx[k], cy[k] = p
    return cx, cy, ct


_HP_WINDOW = os.environ.get("R3D_HP_WINDOW", "1") == "1"
_HP_DELTA = os.environ.get("R3D_HP_DELTA", "1") == "1"

# The colour arguments update_draw passes for the MAIN bar layer. Hoisted so the
# per-frame cache key is an identity compare, not a fresh tuple build. If a skin
# ever makes these vary, compute it per frame instead -- the delta path keys on
# it and falls back to a full recompute on any change (correct, just slower).
_MAIN_COLOUR = ((1.0, 1.0, 1.0), 1.0, GLOW_RGB, GLOW_A)


class _Layer:
    """One distance-field layer (its own size/origin/radius)."""

    def __init__(self, ox, oy, sx, sy, R, k):
        self.ox, self.oy = int(round(ox)), int(round(oy))
        self.sx, self.sy, self.R = sx, sy, R
        self.w_px = int(np.ceil(sx))
        self.h_px = int(np.ceil(sy))
        cx, cy, ct = _build_centerline(sx, sy, R, k)
        self.cx, self.cy, self.ct = cx, cy, ct
        # pixel grid (local layer coords, pixel centres)
        xs = np.arange(self.w_px) + 0.5
        ys = np.arange(self.h_px) + 0.5
        self.X, self.Y = np.meshgrid(xs, ys)            # (h, w)
        # nearest centre-line point per pixel -> d_perp, t_near
        # vectorised over samples in chunks to bound memory
        flatX = self.X.ravel()[:, None]                 # (P,1)
        flatY = self.Y.ravel()[:, None]
        best_d = np.full(flatX.shape[0], 1e18)
        best_t = np.zeros(flatX.shape[0])
        step = 128
        for a in range(0, len(cx), step):
            sl = slice(a, a + step)
            dx = flatX - cx[None, sl]
            dy = flatY - cy[None, sl]
            d2 = dx * dx + dy * dy                       # (P, step)
            idx = np.argmin(d2, axis=1)
            dmin = d2[np.arange(d2.shape[0]), idx]
            upd = dmin < best_d
            best_d[upd] = dmin[upd]
            best_t[upd] = ct[sl][idx[upd]]
        self.d_perp = np.sqrt(best_d).reshape(self.h_px, self.w_px)
        self.t_near = best_t.reshape(self.h_px, self.w_px)
        # horizontal alpha gradient (glow bar: white@0.8 left -> white@1.0 right)
        self.xfrac = (self.X / max(sx, 1.0))

    def window(self, a: float, b: float):
        """Local-coord bbox (y0,y1,x0,x1) containing every pixel whose distance
        to the [a,b] arc is < R. PERF (bit-identical, not an approximation):
        outside it _colour_bar's mixv = clip((R-d)/abs_glow,0,1)**8 is exactly
        0, so alpha is 0, so the ADDITIVE term rgb*255*0 is exactly 0.0 and
        `clip(sub + 0)` returns the uint8 source unchanged. The glow layer is
        572x159 but only ~14% of it is ever visible (a tip cap at the health
        edge), so the other 86% was computed and then added as zero."""
        if b <= a + 1e-6:
            cxs = np.array([np.interp(a, self.ct, self.cx)])
            cys = np.array([np.interp(a, self.ct, self.cy)])
        else:
            m = (self.ct >= a) & (self.ct <= b)
            cxs = np.concatenate([self.cx[m],
                                  [np.interp(a, self.ct, self.cx),
                                   np.interp(b, self.ct, self.cx)]])
            cys = np.concatenate([self.cy[m],
                                  [np.interp(a, self.ct, self.cy),
                                   np.interp(b, self.ct, self.cy)]])
        R = self.R + 1.0                      # margin: d just past R still -> 0
        x0 = max(0, int(np.floor(cxs.min() - R)))
        y0 = max(0, int(np.floor(cys.min() - R)))
        x1 = min(self.w_px, int(np.ceil(cxs.max() + R)))
        y1 = min(self.h_px, int(np.ceil(cys.max() + R)))
        if x1 <= x0 or y1 <= y0:
            return 0, 0, 0, 0
        return y0, y1, x0, x1

    def cap_band(self, b_old: float, b_new: float):
        """Bbox of every pixel whose contribution can differ between arc
        [0,b_old] and [0,b_new]. Pixels left of the band keep d = d_perp
        (independent of the endpoint); pixels right of it are >R from BOTH
        caps, so alpha is 0 in both frames. Covers the centre-line arc
        between the two endpoints so a pixel flipping from body->cap is
        inside it. Bit-identical, not an approximation."""
        lo, hi = (b_old, b_new) if b_old <= b_new else (b_new, b_old)
        m = (self.ct >= lo) & (self.ct <= hi)
        cxs = np.concatenate([self.cx[m], [np.interp(lo, self.ct, self.cx),
                                           np.interp(hi, self.ct, self.cx)]])
        cys = np.concatenate([self.cy[m], [np.interp(lo, self.ct, self.cy),
                                           np.interp(hi, self.ct, self.cy)]])
        R = self.R + 1.0
        x0 = max(0, int(np.floor(cxs.min() - R)))
        y0 = max(0, int(np.floor(cys.min() - R)))
        x1 = min(self.w_px, int(np.ceil(cxs.max() + R)))
        y1 = min(self.h_px, int(np.ceil(cys.max() + R)))
        if x1 <= x0 or y1 <= y0:
            return None
        return y0, y1, x0, x1

    def distance_win(self, a: float, b: float, win) -> np.ndarray:
        """distance() restricted to a window. Elementwise ops commute with
        slicing, so values match the full-layer call exactly."""
        y0, y1, x0, x1 = win
        X = self.X[y0:y1, x0:x1]
        Y = self.Y[y0:y1, x0:x1]
        if b <= a + 1e-6:
            cax = np.interp(a, self.ct, self.cx)
            cay = np.interp(a, self.ct, self.cy)
            return np.sqrt((X - cax) ** 2 + (Y - cay) ** 2)
        d = self.d_perp[y0:y1, x0:x1].copy()
        tn = self.t_near[y0:y1, x0:x1]
        cax = np.interp(a, self.ct, self.cx); cay = np.interp(a, self.ct, self.cy)
        cbx = np.interp(b, self.ct, self.cx); cby = np.interp(b, self.ct, self.cy)
        below = tn < a
        above = tn > b
        d[below] = np.sqrt((X[below] - cax) ** 2 + (Y[below] - cay) ** 2)
        d[above] = np.sqrt((X[above] - cbx) ** 2 + (Y[above] - cby) ** 2)
        return d

    def distance(self, a: float, b: float) -> np.ndarray:
        """Distance to the centre-line restricted to arc-fraction [a,b]
        (rounded caps at the ends — exactly how the shader clips progress)."""
        if b <= a + 1e-6:
            cax = np.interp(a, self.ct, self.cx)
            cay = np.interp(a, self.ct, self.cy)
            return np.sqrt((self.X - cax) ** 2 + (self.Y - cay) ** 2)
        d = self.d_perp.copy()
        cax = np.interp(a, self.ct, self.cx); cay = np.interp(a, self.ct, self.cy)
        cbx = np.interp(b, self.ct, self.cx); cby = np.interp(b, self.ct, self.cy)
        below = self.t_near < a
        above = self.t_near > b
        d[below] = np.sqrt((self.X[below] - cax) ** 2 + (self.Y[below] - cay) ** 2)
        d[above] = np.sqrt((self.X[above] - cbx) ** 2 + (self.Y[above] - cby) ** 2)
        return d


def _colour_bar(d, R, glow_portion, bar_rgb, bar_a, glow_rgb, glow_a):
    """sh_ArgonBarPath.fs getColour: distance -> (rgb(h,w,3), a(h,w))."""
    abs_glow = R * glow_portion
    d = np.clip(d, 0.0, R)
    h, w = d.shape
    rgb = np.empty((h, w, 3), np.float32)
    a = np.empty((h, w), np.float32)
    bar_rgb = np.asarray(bar_rgb, np.float32)
    glow_rgb = np.asarray(glow_rgb, np.float32)
    # region 3: glow falloff (mixValue^8)
    mixv = np.clip((R - d) / max(abs_glow, 1e-6), 0.0, 1.0)
    mixv = mixv * mixv; mixv = mixv * mixv; mixv = mixv * mixv      # ^8
    rgb[:] = glow_rgb
    a[:] = glow_a * mixv
    # region 2: 1px transition mix(glow, bar, factor)
    m2 = d < (R - abs_glow)
    factor = np.clip((R - abs_glow) - d, 0.0, 1.0)
    f = factor[m2][:, None]
    rgb[m2] = glow_rgb[None, :] * (1.0 - f) + bar_rgb[None, :] * f
    a[m2] = glow_a * (1.0 - factor[m2]) + bar_a * factor[m2]
    # region 1: solid bar core
    m1 = d < (R - abs_glow - 1.0)
    rgb[m1] = bar_rgb
    a[m1] = bar_a
    return rgb, a


def _colour_bg(d, R):
    """sh_ArgonBarPathBackground.fs: faint dark->white track + white rim."""
    rel = np.clip(d / R, 0.0, 1.5) / 1.5
    rgb = (np.stack([rel, rel, rel], axis=-1)).astype(np.float32)       # 0..1 grey
    a = (0.2 + 0.6 * rel).astype(np.float32)
    # thin white rim near the outer edge (d in [R-2, R])
    rim = np.clip(d - (R - 1.0), 0.0, 1.0)
    edge = (d > R - 2.0)
    rgb[edge] = rgb[edge] * (rim[edge][:, None]) + 1.0 * (1.0 - rim[edge][:, None])
    fade = (d > R - 1.0)
    a[fade] = a[fade] * (1.0 - rim[fade]) + 1.0 * 0.0   # alpha fades to 0 at very edge
    return rgb, a


class ArgonHealth:
    """Stateful per-render Argon health bar. Call update_draw() each frame."""

    def __init__(self, w: int, h: int, width_frac: float = 0.248,
                 left_frac: float = 0.012, top_frac: float = 0.012):
        self.w, self.h = w, h
        # lazer's default Argon skin pins the health bar to a fixed Width=300
        # (NOT 0.98 relative) — a short top-left bar in the score wedge. We match
        # the reference screenshots' ~0.235*W and scale every lazer absolute
        # constant by k = box_w/300 so the SHAPE (incl. the right-end hook) is
        # an exact scaled copy of lazer's 300px bar.
        box_w = width_frac * w
        k = box_w / DEFAULT_WIDTH
        self.k = k
        box_h = (BAR_HEIGHT + PADDING) * k
        box_ox = left_frac * w
        box_oy = top_frac * h
        R = MAIN_PATH_RADIUS * k
        gR = GLOW_PATH_RADIUS * k
        pad_expand = (GLOW_PATH_RADIUS - MAIN_PATH_RADIUS) * k          # glow container grows out
        self.bg = _Layer(box_ox, box_oy, box_w, box_h, R, k)
        # main has IDENTICAL geometry to bg and _Layer is immutable after
        # construction (distance() copies) — share one instance instead of
        # rebuilding the ~0.24 s distance field (PERF, output-identical).
        self.main = self.bg
        self.glow = _Layer(box_ox - pad_expand, box_oy - pad_expand,
                           box_w + 2 * pad_expand, box_h + 2 * pad_expand, gR, k)
        self._glow_win = None
        self._main_win = None
        self._main_tm = None
        self._main_tm_full = False
        # animation state
        self._hp = None
        self._glow = None
        self._miss_t = None          # ms since miss started, or None
        self._flash_t = 1e9          # ms since last heal flash
        self._prev_hp = None
        # per-frame layer caches (PERF, output-identical): each cache stores
        # the exact (rgb, a) arrays produced for the exact float inputs of the
        # previous frame and is reused only when EVERY input is unchanged —
        # bit-identical by construction. The background track is a pure
        # function of the (fixed) geometry, so it is computed once.
        self._bg_cache = None                      # (rgb, a) — frame-invariant
        self._glow_key = self._glow_cache = None   # keyed (lo, hi, colours, a)
        self._main_key = self._main_cache = None   # keyed hv
        self._orig = (0, 0)                        # crop origin (see clip_box)

    # -- value animation (mirrors ArgonHealthDisplay.Update / miss display) ----
    def _advance(self, target_hp: float, dt_ms: float):
        if self._hp is None:
            self._hp = self._glow = target_hp
            self._prev_hp = target_hp
            return
        # detect a miss (health dropped) -> start/refresh the red drain trail
        if target_hp < self._prev_hp - 1e-4:
            self._miss_t = 0.0
        elif target_hp > self._prev_hp + 1e-4:
            self._flash_t = 0.0          # heal -> brief white glow flash
            if self._miss_t is not None and target_hp >= self._glow:
                self._miss_t = None      # recovered past the trail
        self._prev_hp = target_hp
        self._hp = _damp(self._hp, target_hp, 50.0, dt_ms)
        if self._miss_t is None:
            self._glow = _damp(self._glow, target_hp, 50.0, dt_ms)
        else:
            self._miss_t += dt_ms
            if self._miss_t >= 500.0:    # retract the trail (300ms OutQuint-ish)
                self._glow = _damp(self._glow, target_hp, 80.0, dt_ms)
                if abs(self._glow - target_hp) < 2e-3:
                    self._miss_t = None
            # else: glow frozen (the stretched trail)
        self._flash_t += dt_ms

    def clip_box(self, W: int, H: int):
        """Union of the three layers' pixel rects, clipped to WxH — every
        pixel update_draw can touch. Lets the caller hand update_draw a small
        crop of the frame instead of the whole frame (PERF; the composites
        clip identically inside the crop, so output is unchanged)."""
        layers = (self.bg, self.main, self.glow)
        xa = max(0, min(l.ox for l in layers))
        ya = max(0, min(l.oy for l in layers))
        xb = min(W, max(l.ox + l.w_px for l in layers))
        yb = min(H, max(l.oy + l.h_px for l in layers))
        return xa, ya, xb, yb

    # PERF (output-identical): the per-frame composites used to rebuild the
    # colour terms — (1-a), rgb*255*a — from the cached (rgb, a) layer EVERY
    # frame. Those terms are pure functions of the cached layer, so they are
    # now precomputed ONCE when a layer cache is (re)filled (_prep_add /
    # _prep_over) and the composite is a single fused multiply-add over the
    # crop. Elementwise ops commute with slicing, so the values are
    # bit-identical to the old per-frame computation.

    @staticmethod
    def _prep_add(rgb, a):
        """Precompute the additive term rgb*255*a (broadcast-ready)."""
        return (rgb * 255.0) * a[..., None]

    @staticmethod
    def _prep_over(rgb, a):
        """Precompute (1-a, rgb*255*a) for the src-over composite."""
        ab = a[..., None]
        return 1.0 - ab, (rgb * 255.0) * ab

    def _composite_additive(self, img, layer, tm, win=None, tm_full=False):
        if win is None:
            wx0, wy0, lw, lh = 0, 0, layer.w_px, layer.h_px
        else:
            wy0, wy1, wx0, wx1 = win
            lw, lh = wx1 - wx0, wy1 - wy0
            if lw <= 0 or lh <= 0:
                return
        x0 = layer.ox + wx0 - self._orig[0]
        y0 = layer.oy + wy0 - self._orig[1]
        H, W = img.shape[:2]
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W, x0 + lw), min(H, y0 + lh)
        if xb <= xa or yb <= ya:
            return
        lx0, ly0 = xa - x0, ya - y0
        if tm_full and win is not None:
            ly0 += wy0
            lx0 += wx0
        sub = img[ya:yb, xa:xb].astype(np.float32)
        tms = tm[ly0:ly0 + (yb - ya), lx0:lx0 + (xb - xa)]
        img[ya:yb, xa:xb] = np.clip(sub + tms, 0, 255).astype(np.uint8)

    def _composite_over(self, img, layer, om, tm):
        x0, y0 = layer.ox - self._orig[0], layer.oy - self._orig[1]
        H, W = img.shape[:2]
        xa, ya = max(0, x0), max(0, y0)
        xb, yb = min(W, x0 + layer.w_px), min(H, y0 + layer.h_px)
        if xb <= xa or yb <= ya:
            return
        lx0, ly0 = xa - x0, ya - y0
        sub = img[ya:yb, xa:xb].astype(np.float32)
        oms = om[ly0:ly0 + (yb - ya), lx0:lx0 + (xb - xa)]
        tms = tm[ly0:ly0 + (yb - ya), lx0:lx0 + (xb - xa)]
        img[ya:yb, xa:xb] = np.clip(sub * oms + tms, 0, 255).astype(np.uint8)

    def update_draw(self, rgb_arr: np.ndarray, hp: float, dt_ms: float,
                    origin=(0, 0)):
        """Advance the animation by dt_ms and composite the bar onto rgb_arr
        (HxWx3 uint8, modified in place). `origin` = rgb_arr's top-left in
        screen px when the caller passes a clip_box() crop instead of the
        whole frame (default (0,0) keeps the old full-frame contract)."""
        self._orig = origin
        hp = float(max(0.0, min(1.0, hp)))
        self._advance(hp, dt_ms)
        hv = float(max(0.0, min(1.0, self._hp)))
        gv = float(max(0.0, min(1.0, self._glow)))

        # 1) background track (full length, normal blend) — geometry-only,
        # identical every frame: compute once, composite the cached layer.
        if self._bg_cache is None:
            bd = self.bg.distance(0.0, 1.0)
            self._bg_cache = self._prep_over(*_colour_bg(bd, self.bg.R))
        self._composite_over(rgb_arr, self.bg, *self._bg_cache)

        # 2) glow bar over [hv, max(gv,hv)] (additive)
        lo, hi = hv, max(gv, hv)
        # miss colouring of the trail
        if self._miss_t is not None:
            p = min(1.0, self._miss_t / 300.0)
            bar_rgb = tuple((1.0 - p) * 1.0 + p * c for c in RED_BAR_RGB)
            glow_rgb = RED_GLOW_RGB
            glow_a = GLOW_A + (1.0 - GLOW_A) * 0.4
        else:
            bar_rgb = (1.0, 1.0, 1.0)
            glow_rgb = GLOW_RGB
            # brief white flash on heal
            if self._flash_t < 300.0:
                k = 1.0 - self._flash_t / 300.0
                glow_rgb = tuple(GLOW_RGB[i] + (1.0 - GLOW_RGB[i]) * k for i in range(3))
            glow_a = GLOW_A
        if hv > 1e-4:
            # always draw the glow (zero-length -> a bright tip cap at the
            # leading health edge, as in lazer); a miss stretches it into the
            # red drain trail. The layer is a pure function of the key below
            # (the damped values converge, so stable stretches reuse it).
            gkey = (lo, hi, bar_rgb, glow_rgb, glow_a)
            if gkey != self._glow_key:
                gwin = self.glow.window(lo, hi) if _HP_WINDOW else None
                if gwin is None:
                    gd = self.glow.distance(lo, hi)
                    gxf = self.glow.xfrac
                else:
                    gy0, gy1, gx0, gx1 = gwin
                    gd = self.glow.distance_win(lo, hi, gwin)
                    gxf = self.glow.xfrac[gy0:gy1, gx0:gx1]
                grgb, ga = _colour_bar(gd, self.glow.R, GLOW_GLOW_PORTION,
                                       bar_rgb, 1.0, glow_rgb, glow_a)
                ga = ga * (0.8 + 0.2 * gxf)                # horizontal gradient
                self._glow_key = gkey
                self._glow_win = gwin
                self._glow_cache = self._prep_add(grgb, ga)
            self._composite_additive(rgb_arr, self.glow, self._glow_cache,
                                     self._glow_win)

        # 3) main bar over [0, hv] (additive, white core + blue edge) — pure
        # function of hv; reuse while hv is unchanged.
        if hv > 1e-4:
            # The delta buffer is only valid while the main bar's COLOURS are
            # unchanged. They are constant today, but skins and configurable HUD
            # layouts are coming -- a skin that tints the bar or ties its colour
            # to HP would otherwise leave stale pixels in the buffer with no
            # error. Key on the colour tuple and fall back to a full recompute
            # whenever it moves, so a future skin degrades in SPEED, not
            # correctness.
            mcol = _MAIN_COLOUR
            mkey = (hv, mcol)
            if mkey != self._main_key or self._main_cache is None:
                _prev = self._main_key
                mwin = self.main.window(0.0, hv) if _HP_WINDOW else None
                # INCREMENTAL (bit-identical): the main bar's colours are
                # CONSTANT -- white core + GLOW_RGB edge, no miss/flash term --
                # so hv is the only variable and only a ~32x20 band around the
                # moving cap can differ (measured: 1.2% of the layer, flat at
                # every HP level). Keep the additive term in a persistent
                # full-layer buffer and rewrite just that band.
                band = None
                if (_HP_DELTA and self._main_tm is not None
                        and _prev is not None and _prev[1] == mcol):
                    band = self.main.cap_band(_prev[0], hv)
                if band is not None:
                    by0, by1, bx0, bx1 = band
                    bd = self.main.distance_win(0.0, hv, band)
                    brgb, ba = _colour_bar(bd, self.main.R, MAIN_GLOW_PORTION,
                                           (1.0, 1.0, 1.0), 1.0, GLOW_RGB, GLOW_A)
                    self._main_tm[by0:by1, bx0:bx1] = self._prep_add(brgb, ba)
                    self._main_key = mkey
                    self._main_win = mwin
                    self._main_cache = self._main_tm
                    self._main_tm_full = True
                    md = None
                else:
                    md = self.main.distance(0.0, hv)
                    mrgb, ma = _colour_bar(md, self.main.R, MAIN_GLOW_PORTION,
                                           (1.0, 1.0, 1.0), 1.0, GLOW_RGB, GLOW_A)
                    self._main_tm = self._prep_add(mrgb, ma)
                    self._main_key = mkey
                    self._main_win = mwin
                    self._main_cache = self._main_tm
                    self._main_tm_full = True
                    md = None
            self._composite_additive(rgb_arr, self.main, self._main_cache,
                                     self._main_win, self._main_tm_full)
