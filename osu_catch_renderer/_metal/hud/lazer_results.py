"""LAZER RESULTS SCREEN for osu!catch — the osu!(lazer) ranking screen, ported
from the osu!STANDARD renderer's render/lazer_results.py (which itself ports
osu.Game/Screens/Ranking/* — MIT, ppy/osu, no ppy assets; every texture is
procedurally baked; font = the same bundled Nunito, the lazer-Torus stand-in).

WHAT IS PORTED 1:1 from the std module (same constants, same geometry, same
draw code, adapted from GL sprites to catch's CPU/PIL compositor):

  * the AccuracyCircle — the thick achieved-accuracy arc with the FIXED
    vertical CYAN→GREEN gradient (#7CF6FF→#BAFFA9) + sweeping tip dash; the
    dim gray background ring; the THIN inner GradedCircles rank ring with the
    OsuColour.ForRank bands (D ff5a5a, C ff8e5d, B e3b130, A 88da20,
    S 02b5c3, X/SS de31ae), GRADE_SPACING notches and the virtual-SS slice;
    the six RankBadge pills OUTSIDE at their Interpolation.Lerp visual
    positions; the WHITE centre rank letter with the rank-coloured glow that
    punches in at the end of the sweep (bake_accuracy_base / bake_accuracy_arc
    / _badge_pill / bake_grade_letter — verbatim ports).
  * the rounded featured panel (560×940 virtual): avatar + name header, map
    title + artist, the rolling score (Nunito Light), the played-mods badge
    row, the star-rating pill (ForStarDifficulty colour + procedural star) +
    diff name + "mapped by", the ACCURACY / MAX COMBO / PP row, the judgment
    row, the "Played on <date>" footer.
  * the stage-1 timeline: panel fade (320ms), arc sweep (260ms delay, 1150ms
    OutCubic), grade punch (340ms, 1.42→1.0), score roll, and the flank
    leaderboard cards' staggered outward slide-in (300ms + 80ms/card, 460ms
    OutQuint, 96 virtual px).

CATCH ADAPTATIONS (owner spec, honest + documented):
  * grade/ring cutoffs are CATCH's (lazer CatchScoreProcessor / wiki, the same
    cutoffs hud._catch_grade uses): SS=100%, S≥98, A≥94, B≥90, C≥85, D<85 —
    in lazer the AccuracyCircle reads the RULESET's accuracy cutoffs, so the
    ring bands + badge Lerp positions are recomputed from these (faithful,
    not a deviation). Catch's tight top bands crowd the B/A/S/SS badges, so
    the badge pills are slightly smaller + slightly further out than std's
    (0.072/0.046 @ r 0.445 vs 0.088/0.050 @ r 0.435) to keep S/SS legible.
  * judgments are catch's: FRUIT / DROP / DROPLET / MISS (MISS = count_miss
    only — tiny-droplet misses are not misses in either client; they show as
    the DROPLET cell's caught/total instead),
    coloured with the same palette as the flank cards (lb_cards.RESULT_COLORS)
    so the whole screen reads as one UI. No slider tick/end rows — catch has
    no such concept (std's grid_c is omitted).
  * stars/pp come from rosu-pp (converted to the Catch ruleset, the
    scene.compute_pp_curve path); fail-soft to "--".
  * stage 2 — the panel slides LEFT while three stats panels unfold from the
    right, staggered (STAGE1_MS=2000, OPEN_MS=900 OutQuint, STAGGER_MS=160 —
    std's exact timeline/motion) — but populated with CATCH's real data, not
    std's hit-error graphs (catch has no hit-error deltas or aim points, so
    std's Timing Distribution + Accuracy heatmap CANNOT be populated and are
    not faked):
      1. PERFORMANCE — accuracy / max-combo-vs-map-max / achieved-vs-SS-pp
         bars + the "Achieved X pp / Maximum Y pp" footer (rosu-pp, the same
         Catch-ruleset conversion as the pp row; fail-soft to fewer rows).
      2. COMBO — the combo-progression area chart straight from the catch
         sim's per-object checkpoints (the running combo the HUD showed),
         with red ticks at each combo break; falls back to the rosu-pp
         MOVEMENT STRAIN curve (map difficulty over time) when the sim isn't
         plumbed through, and to an honest "timeline unavailable" footer when
         neither exists.
      3. CATCHES — per-type caught/total bars (FRUIT / DROP / DROPLET, play
         counts vs the map's object counts, the flank-card palette) + the
         hyperdash-usage footer (dashes used from the sim's hyper windows /
         hyperdash fruits in the map, plus the banana count).
  * the scene under the screen goes to BLACK (the results-fade opacity drives
    a full black wash, matching std where the gameplay has already faded to
    black by results_start and the screen dims what's left).
  * flank cards + "#N on <map>" banner are catch's existing lb_cards bakes
    (kept, per owner), laid out around the featured panel exactly like std:
    featured centred, ranked neighbours marching outward from its edges.

Everything is laid out in the same 1080-height virtual space as std
(k = H / 1080). Static elements bake once; the accuracy arc and the rolling
score re-bake only while they change; once the animation settles the full
frame is cached and reused for the hold.
"""
from __future__ import annotations

import math
import os
from datetime import datetime

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from osu_catch_renderer._metal.hud.fonts import font as _catch_font
from osu_catch_renderer._metal.hud.lb_cards import (DB_PATH, RESULT_COLORS as CATCH_RESULT_COLORS,
                       bake_round_panel)

UH = 1080.0                    # virtual design height (std convention)

# --- timeline (ms from results start) — std lazer_results values ---------------------
FADE_MS = 320.0                # panel opacity ramp
SWEEP_DELAY_MS = 260.0         # accuracy arc sweep start
SWEEP_MS = 1150.0              # AccuracyCircle ACCURACY_TRANSFORM_DURATION (scaled)
BADGE_MS = 340.0               # rank badge (grade letter) punch duration
LB_SLIDE_START_MS = 300.0      # flank-card entrance begins
LB_SLIDE_STAGGER_MS = 80.0     # per-rank stagger (outer cards arrive later)
LB_SLIDE_MS = 460.0            # per-card slide-in duration (OutQuint)
LB_SLIDE_OFFSET = 96.0         # virtual px each card travels inward
# stage 2 (std lazer_results values): the featured panel slides LEFT while
# the stats panels unfold from the right, staggered
STAGE1_MS = 2000.0             # panel settles + holds until here, then opens
OPEN_MS = 900.0                # panel slide + stats unfold (OutQuint)
STAGGER_MS = 160.0             # per-stats-panel unfold stagger
SETTLE_MS = 3800.0             # everything at rest → the frame is cached
                               # (std MIN_TOTAL_MS; last unfold ends ~3340)

# featured panel geometry (virtual px) — std lazer_results values
PANEL_W = 560.0
PANEL_H = 940.0
ACC_DISP = 380.0               # accuracy-circle canvas display size

VIRTUAL_SS_PERCENTAGE = 0.01   # AccuracyCircle: the reserved SS notch

# rank accuracy thresholds — CATCH's cutoffs (lazer CatchScoreProcessor /
# osu! wiki; the same cutoffs hud._catch_grade uses). In lazer the
# AccuracyCircle reads the ruleset's cutoffs, so the ring bands and badge
# positions derive from THESE (std's are the osu!std cutoffs).
RANK_THRESHOLDS = [
    (0.00, "D"),
    (0.85, "C"),
    (0.90, "B"),
    (0.94, "A"),
    (0.98, "S"),
    (1.00, "SS"),
]


def _hex(s: str) -> tuple[float, float, float]:
    s = s.lstrip("#")
    return (int(s[0:2], 16) / 255.0, int(s[2:4], 16) / 255.0,
            int(s[4:6], 16) / 255.0)


# lazer's OsuColour.ForRank — the EXACT hexes std's port uses
# (osu.Game/Graphics/OsuColour.cs ForRank).
FOR_RANK = {
    "D": _hex("ff5a5a"),
    "C": _hex("ff8e5d"),
    "B": _hex("e3b130"),
    "A": _hex("88da20"),
    "S": _hex("02b5c3"),
    "SS": _hex("de31ae"),
    "F": _hex("ff5a5a"),
}
FOR_RANK["X"] = FOR_RANK["SS"]
FOR_RANK["SSH"] = FOR_RANK["SS"]
FOR_RANK["XH"] = FOR_RANK["SS"]
FOR_RANK["SH"] = FOR_RANK["S"]

GRADE_SPACING_PERCENTAGE = 2.0 / 360.0

# the achieved-accuracy arc's FIXED vertical gradient (NOT rank-coloured)
ARC_GRAD_TOP = _hex("7CF6FF")
ARC_GRAD_BOT = _hex("BAFFA9")

# accuracy-circle geometry as a fraction of the (square) bake canvas S —
# std's values, with the badge pills slightly smaller + further out because
# catch's B/A/S/SS bands crowd the top of the ring (see module docstring).
ACC_ARC_R = 0.350
ACC_ARC_W = 0.066
ACC_GRAD_R = 0.293
ACC_GRAD_W = 0.020
ACC_BADGE_R = 0.445            # std: 0.435
ACC_BADGE_W = 0.072            # std: 0.088
ACC_BADGE_H = 0.046            # std: 0.050

# results-screen text weights (Nunito variable `wght`, the lazer Torus
# stand-in — same as std)
RESULTS_TEXT_WEIGHT = 500      # Nunito Medium (body/labels)
RESULTS_SCORE_WEIGHT = 330     # Nunito Light (the big rolling score)

TEXT_SS = 3                    # supersample factors (std's values)
SHAPE_SS = 2

# mod badges (std's results-panel values)
MOD_PILL_VH = 34.0
MOD_PILL_TEXT_VPX = 19.0
MOD_PILL_GAP_V = 8.0
MOD_PILL_ALPHA = 235

# osu! mod bitmask → display names (NC eats DT, PF eats SD) — std results.py
_MOD_BITS = ((2, "EZ"), (1, "NF"), (256, "HT"), (8, "HD"), (16, "HR"),
             (16384, "PF"), (32, "SD"), (512, "NC"), (64, "DT"),
             (1024, "FL"), (128, "RX"), (8192, "AP"), (4096, "SO"),
             (4, "TD"), (536870912, "V2"))
_MOD_REDUCTION = {"EZ", "NF", "HT", "SO"}
_MOD_AUTOMATION = {"RX", "AP", "V2", "TD"}
MOD_COLOR_REDUCTION = (0.45, 0.78, 0.36)
MOD_COLOR_INCREASE = (0.90, 0.32, 0.42)
MOD_COLOR_AUTOMATION = (0.36, 0.62, 0.92)

# OsuColour.ForStarDifficulty gradient stops (star, hex) — std's port
STAR_SPECTRUM = [
    (0.1, "aaaaaa"), (0.1, "4290fb"), (1.25, "4fc0ff"), (2.0, "4fffd5"),
    (2.5, "7cff4f"), (3.3, "f6f05c"), (4.2, "ff8068"), (4.9, "ff3c71"),
    (5.8, "6563de"), (6.7, "18158e"), (7.7, "000000"), (9.0, "000000"),
]

# the bundled Nunito (copied from the std renderer's assets — SIL OFL)
NUNITO_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "assets", "fonts", "Nunito[wght].ttf"))


def _nunito_loader(weight: int):
    """A (px)->font loader for the bundled Nunito at a variable `wght`
    (std's lazer-Torus stand-in). Falls back to catch's resolver if the
    asset is missing so a stripped checkout still renders."""
    def _load(px: int):
        try:
            f = ImageFont.truetype(NUNITO_PATH, max(int(px), 6))
        except OSError:
            return _catch_font(max(int(px), 6))
        try:
            f.set_variation_by_axes([weight])
        except Exception:  # noqa: BLE001 — non-variable build → default face
            pass
        return f
    return _load


_font_body = _nunito_loader(RESULTS_TEXT_WEIGHT)
_font_score = _nunito_loader(RESULTS_SCORE_WEIGHT)


# --- pure helpers (std ports) --------------------------------------------------------

def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def ease_out_quint(p: float) -> float:
    p = _clamp01(p)
    return 1.0 - (1.0 - p) ** 5


def ease_out_cubic(p: float) -> float:
    p = _clamp01(p)
    return 1.0 - (1.0 - p) ** 3


def catch_grade(acc_frac: float) -> str:
    """osu!catch grade from accuracy [0..1] — the RANK_THRESHOLDS cutoffs
    (same result as hud._catch_grade; local to avoid a circular import)."""
    acc_frac = _clamp01(acc_frac)
    if acc_frac >= 1.0:
        return "SS"
    grade = "D"
    for lo, g in RANK_THRESHOLDS[:-1]:
        if acc_frac >= lo:
            grade = g
    return grade


def rank_ring_bands() -> list[tuple[float, float, str]]:
    """[(acc_lo, acc_hi, grade)] — lazer's GradedCircles band layout with
    CATCH's cutoffs: the S band stops at 1 − VIRTUAL_SS_PERCENTAGE and a
    distinct SS/X band owns the final slice (the virtual-SS notch)."""
    ss_lo = 1.0 - VIRTUAL_SS_PERCENTAGE
    bands = []
    for i, (lo, g) in enumerate(RANK_THRESHOLDS[:-1]):
        hi = RANK_THRESHOLDS[i + 1][0]
        if g == "S":
            hi = ss_lo
        bands.append((lo, hi, g))
    bands.append((ss_lo, 1.00, "SS"))
    return bands


def rank_badge_positions() -> list[tuple[float, str]]:
    """[(visual_acc, grade)] — lazer AccuracyCircle.cs RankBadge placement
    (each badge at its band's Interpolation.Lerp VISUAL position), computed
    from CATCH's cutoffs:

        RankBadge(accuracyD, Lerp(accuracyD, accuracyC, 0.5),  D)
        RankBadge(accuracyC, Lerp(accuracyC, accuracyB, 0.5),  C)
        RankBadge(accuracyB, Lerp(accuracyB, accuracyA, 0.5),  B)
        RankBadge(accuracyA, Lerp(accuracyA, accuracyS, 0.25), A)
        RankBadge(accuracyS, Lerp(accuracyS, accuracyX−VSS, 0.25), S)
        RankBadge(accuracyX, accuracyX, X/SS)
    """
    d, c, b, a, s, x = (t for t, _g in RANK_THRESHOLDS)
    ss_lo = x - VIRTUAL_SS_PERCENTAGE
    return [
        (_lerp(d, c, 0.5), "D"),
        (_lerp(c, b, 0.5), "C"),
        (_lerp(b, a, 0.5), "B"),
        (_lerp(a, s, 0.25), "A"),
        (_lerp(s, ss_lo, 0.25), "S"),
        (x, "SS"),
    ]


def for_star_difficulty(stars: float) -> tuple[float, float, float]:
    """The star-rating pill colour — OsuColour.ForStarDifficulty (std port)."""
    s = max(float(stars), 0.0)
    if s <= STAR_SPECTRUM[0][0]:
        return _hex(STAR_SPECTRUM[0][1])
    for i in range(1, len(STAR_SPECTRUM)):
        s0, h0 = STAR_SPECTRUM[i - 1]
        s1, h1 = STAR_SPECTRUM[i]
        if s <= s1:
            t = 0.0 if s1 == s0 else (s - s0) / (s1 - s0)
            c0, c1 = _hex(h0), _hex(h1)
            return (_lerp(c0[0], c1[0], t), _lerp(c0[1], c1[1], t),
                    _lerp(c0[2], c1[2], t))
    return _hex(STAR_SPECTRUM[-1][1])


def target_arc_value(acc_frac: float, grade: str) -> float:
    """AccuracyCircle target fill: a true SS closes the ring, everything else
    caps at 1 − VIRTUAL_SS_PERCENTAGE so the SS notch stays open."""
    acc_frac = _clamp01(acc_frac)
    if grade in ("SS", "X", "SSH", "XH"):
        return acc_frac
    return min(1.0 - VIRTUAL_SS_PERCENTAGE, acc_frac)


def acc_to_angle_deg(acc: float) -> float:
    """Accuracy [0,1] → PIL degrees from 3 o'clock clockwise, ring starting
    at 12 o'clock."""
    return 270.0 + _clamp01(acc) * 360.0


def mods_string(mods: int) -> str:
    """Comma-joined mod acronyms of an .osr bitmask; NC/PF absorb DT/SD.
    '' for nomod (std results.py port)."""
    if mods & 512:
        mods &= ~64
    if mods & 16384:
        mods &= ~32
    return ",".join(name for bit, name in _MOD_BITS if mods & bit)


def mod_pill_color(acr: str) -> tuple[float, float, float]:
    if acr in _MOD_REDUCTION:
        return MOD_COLOR_REDUCTION
    if acr in _MOD_AUTOMATION:
        return MOD_COLOR_AUTOMATION
    return MOD_COLOR_INCREASE


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n - 1] + "…"


def _bake_width(font, text: str, px: int) -> int:
    """The exact pixel width bake_text() produces (std port)."""
    if not text:
        return 1
    try:
        x0, _y0, x1, _y1 = font.getbbox(text)
    except AttributeError:
        w0, _h0 = font.getsize(text)          # type: ignore[attr-defined]
        x0, x1 = 0, w0
    pad = max(int(px) // 12, 2)
    return max(x1 - x0, 1) + 2 * pad


def _ellipsize(font, text: str, max_w: float, px: int) -> str:
    if _bake_width(font, text, px) <= max_w:
        return text
    while len(text) > 1:
        text = text[:-1]
        cand = text.rstrip() + "…"
        if _bake_width(font, cand, px) <= max_w:
            return cand
    return "…"


# --- procedural bakes (PIL) — std ports, returning PIL images ------------------------

def bake_text(text: str, px: int, color, loader=_font_body,
              ss: int = TEXT_SS) -> Image.Image:
    """One baked text line → PIL RGBA. Supersampled at px*ss then LANCZOS-
    downscaled (std's bake_text). Blank text → 1×1 stub."""
    if not text:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))
    px = max(int(px), 6)
    font = loader(px)
    try:
        x0, y0, x1, y1 = font.getbbox(text)
    except AttributeError:
        w0, h0 = font.getsize(text)          # type: ignore[attr-defined]
        x0, y0, x1, y1 = 0, 0, w0, h0
    pad = max(px // 12, 2)
    w = max(x1 - x0, 1) + 2 * pad
    h = max(y1 - y0, 1) + 2 * pad
    rgb = tuple(int(round(c * 255)) for c in color)
    ss = max(int(ss), 1)
    if ss == 1:
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font,
                                 fill=(*rgb, 255))
        return img
    fb = loader(px * ss)
    try:
        bx0, by0, bx1, by1 = fb.getbbox(text)
    except AttributeError:
        bw0, bh0 = fb.getsize(text)          # type: ignore[attr-defined]
        bx0, by0, bx1, by1 = 0, 0, bw0, bh0
    padb = pad * ss
    Wb = max(bx1 - bx0, 1) + 2 * padb
    Hb = max(by1 - by0, 1) + 2 * padb
    big = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    ImageDraw.Draw(big).text((padb - bx0, padb - by0), text, font=fb,
                             fill=(*rgb, 255))
    return big.resize((w, h), Image.LANCZOS)


def fit_text(text: str, px_virtual: float, color, max_w_virtual: float,
             k: float, min_px_virtual: float | None = None,
             loader=_font_body) -> Image.Image:
    """Bake a text row SIZED TO FIT max_w_virtual (std's _fit_text): shrink
    the font toward min_px_virtual until it fits, then ellipsis-truncate."""
    text = text or ""
    if min_px_virtual is None:
        min_px_virtual = max(px_virtual * 0.62, 12.0)
    max_w = max_w_virtual * k
    px = float(px_virtual)
    step = max((px_virtual - min_px_virtual) / 12.0, 1.0)
    chosen = min_px_virtual
    while px >= min_px_virtual:
        fpx = max(int(round(px * k)), 6)
        if _bake_width(loader(fpx), text, fpx) <= max_w:
            chosen = px
            break
        px -= step
    fpx = max(int(round(chosen * k)), 6)
    font = loader(fpx)
    if _bake_width(font, text, fpx) > max_w:
        text = _ellipsize(font, text, max_w, fpx)
    return bake_text(text, fpx, color, loader)


def _hsv(h: float, s: float, v: float) -> tuple[float, float, float]:
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


def _name_hash(name: str) -> int:
    import hashlib
    key = (name or "?").strip().lower().encode("utf-8", "ignore")
    return int(hashlib.md5(key).hexdigest(), 16)


def _avatar_initials(name: str) -> str:
    import re
    name = (name or "").strip()
    if not name:
        return "?"
    tokens = [t for t in re.split(r"[\s_.\-]+", name) if t]
    if len(tokens) >= 2:
        return (tokens[0][0] + tokens[1][0]).upper()
    return tokens[0][0].upper()


def bake_avatar(px: int, name: str, avatar_bytes: bytes | None = None
                ) -> Image.Image:
    """The featured-card CIRCULAR avatar chip (std's bake_avatar): Discord PNG
    bytes cover-fit + disc-clipped + soft ring when available, else the
    deterministic username-hued disc with centred initials. Never raises."""
    px = max(int(px), 8)
    h = _name_hash(name)
    hue = (h % 360) / 360.0
    sat = 0.42 + ((h >> 9) % 18) / 100.0
    if avatar_bytes:
        try:
            from io import BytesIO
            src = Image.open(BytesIO(avatar_bytes)).convert("RGBA")
            sw, sh = src.size
            scale = px / max(min(sw, sh), 1)
            nw = max(int(sw * scale + 0.5), px)
            nh = max(int(sh * scale + 0.5), px)
            src = src.resize((nw, nh), Image.LANCZOS)
            lft, top = (nw - px) // 2, (nh - px) // 2
            img = src.crop((lft, top, lft + px, top + px))
            mask = Image.new("L", (px, px), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, px - 1, px - 1], fill=255)
            img.putalpha(mask)
            d = ImageDraw.Draw(img)
            ring = _hsv(hue, max(sat - 0.16, 0.0), 0.92)
            rc = tuple(int(round(c * 255)) for c in ring)
            lw = max(int(px * 0.045), 2)
            off = lw * 0.5
            d.ellipse([off, off, px - 1 - off, px - 1 - off],
                      outline=(*rc, 150), width=lw)
            return img
        except Exception:  # noqa: BLE001 — corrupt/animated → procedural chip
            pass
    c0 = _hsv(hue, sat, 0.62)
    c1 = _hsv((hue + 0.06) % 1.0, min(sat + 0.08, 1.0), 0.34)
    img = Image.new("RGBA", (px, px), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for y in range(px):
        f = y / max(px - 1, 1)
        col = tuple(int(round(_lerp(c0[i], c1[i], f) * 255)) for i in range(3))
        d.line([(0, y), (px, y)], fill=(*col, 255))
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, px - 1, px - 1], fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    ring = _hsv(hue, max(sat - 0.16, 0.0), 0.92)
    rc = tuple(int(round(c * 255)) for c in ring)
    lw = max(int(px * 0.045), 2)
    off = lw * 0.5
    d.ellipse([off, off, px - 1 - off, px - 1 - off],
              outline=(*rc, 150), width=lw)
    ini = _avatar_initials(name)
    font = _font_body(int(px * (0.46 if len(ini) >= 2 else 0.56)))
    try:
        x0, y0, x1, y1 = font.getbbox(ini)
    except AttributeError:
        x1, y1 = font.getsize(ini); x0 = y0 = 0     # type: ignore
    d.text(((px - (x1 - x0)) / 2 - x0, (px - (y1 - y0)) / 2 - y0), ini,
           font=font, fill=(255, 255, 255, 240))
    return img


def bake_grade_letter(text: str, px: int, fill, glow, loader=_font_body,
                      ss: int = TEXT_SS) -> Image.Image:
    """The AccuracyCircle centre rank letter (std port): WHITE fill with a
    soft rank-coloured outer glow (lazer DrawableRank's coloured EdgeEffect).
    Supersampled then LANCZOS-downscaled."""
    px = max(int(px), 8)
    ss = max(int(ss), 1)
    font = loader(px)
    try:
        x0, y0, x1, y1 = font.getbbox(text)
    except AttributeError:
        x1, y1 = font.getsize(text); x0 = y0 = 0     # type: ignore
    tw = max(x1 - x0, 1)
    th = max(y1 - y0, 1)
    pad = max(int(px * 0.42), 8)
    W = tw + 2 * pad
    H = th + 2 * pad
    bpx = px * ss
    fb = loader(bpx)
    try:
        bx0, by0, bx1, by1 = fb.getbbox(text)
    except AttributeError:
        bx1, by1 = fb.getsize(text); bx0 = by0 = 0   # type: ignore
    btw = max(bx1 - bx0, 1)
    bth = max(by1 - by0, 1)
    padb = pad * ss
    Wb = btw + 2 * padb
    Hb = bth + 2 * padb
    ox, oy = padb - bx0, padb - by0
    gc = tuple(int(round(c * 255)) for c in glow)
    glyph = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    ImageDraw.Draw(glyph).text((ox, oy), text, font=fb, fill=(*gc, 255))
    tight = glyph.filter(ImageFilter.GaussianBlur(max(bpx * 0.045, 1)))
    wide = glyph.filter(ImageFilter.GaussianBlur(max(bpx * 0.11, 2)))
    out = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    for layer in (wide, wide, tight, tight):          # stack → a bright halo
        out = Image.alpha_composite(out, layer)
    fc = tuple(int(round(c * 255)) for c in fill)
    ImageDraw.Draw(out).text((ox, oy), text, font=fb, fill=(*fc, 255))
    if ss != 1:
        out = out.resize((W, H), Image.LANCZOS)
    return out


def bake_star(px: int, color) -> Image.Image:
    """A filled 5-point star sprite (std port — the font has no ★ glyph)."""
    S = max(int(px), 8)
    ss = 4
    big = Image.new("RGBA", (S * ss, S * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(big)
    cx = cy = S * ss / 2.0
    r_out = S * ss * 0.5
    r_in = r_out * 0.42
    pts = []
    for i in range(10):
        r = r_out if i % 2 == 0 else r_in
        ang = math.radians(-90 + i * 36)
        pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
    col = tuple(int(round(c * 255)) for c in color)
    d.polygon(pts, fill=(*col, 255))
    return big.resize((S, S), Image.LANCZOS)


def bake_accuracy_base(px: int, ss: int = SHAPE_SS) -> Image.Image:
    """The AccuracyCircle background, the std 1:1 port (osu.Game/Screens/
    Ranking/Expanded/Accuracy/{AccuracyCircle,GradedCircles,RankBadge}.cs):
    dim gray background ring + the THIN inner GradedCircles ForRank ring with
    the GRADE_SPACING notches + the six RankBadge pills OUTSIDE at their Lerp
    visual positions — with CATCH's cutoffs driving the bands/positions.
    Baked once; the cyan→green achieved arc is a separate bake drawn over."""
    S_target = max(int(px), 64)
    ss = max(int(ss), 1)
    S = S_target * ss
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx = cy = S / 2.0
    half_gap = GRADE_SPACING_PERCENTAGE / 2.0
    # dim gray "Background circle" (OsuColour.Gray(47), alpha 0.5), full ring
    R = ACC_ARC_R * S
    W = max(int(round(ACC_ARC_W * S)), 2)
    d.arc([cx - R, cy - R, cx + R, cy + R], 0, 360, fill=(47, 47, 47, 128),
          width=W)
    # thin inner GradedCircles rank ring (ForRank bands + boundary notches)
    Rg = ACC_GRAD_R * S
    Wg = max(int(round(ACC_GRAD_W * S)), 2)
    gbox = [cx - Rg, cy - Rg, cx + Rg, cy + Rg]
    for lo, hi, g in rank_ring_bands():
        col = tuple(int(round(c * 255)) for c in FOR_RANK[g])
        a0 = acc_to_angle_deg(lo + half_gap)
        a1 = acc_to_angle_deg(hi - half_gap)
        if a1 > a0:
            d.arc(gbox, a0, a1, fill=(*col, 255), width=Wg)
    # RankBadge pills at their Lerp visual positions, outside the ring
    for vis, g in rank_badge_positions():
        ang = math.radians(acc_to_angle_deg(vis))
        bx = cx + ACC_BADGE_R * S * math.cos(ang)
        by = cy + ACC_BADGE_R * S * math.sin(ang)
        _badge_pill(img, bx, by, S, g)
    if ss != 1:
        img = img.resize((S_target, S_target), Image.LANCZOS)
    return img


def _badge_pill(img, cx, cy, S, grade) -> None:
    """One RankBadge: a rounded pill in the rank's ForRank colour with the
    rank letter + soft drop shadow (std port)."""
    d = ImageDraw.Draw(img)
    w = ACC_BADGE_W * S
    h = ACC_BADGE_H * S
    col = tuple(int(round(c * 255)) for c in FOR_RANK.get(grade,
                                                          (0.8, 0.8, 0.85)))
    sh = max(h * 0.10, 1.0)
    d.rounded_rectangle([cx - w / 2, cy - h / 2 + sh, cx + w / 2,
                         cy + h / 2 + sh], radius=h / 2, fill=(0, 0, 0, 70))
    d.rounded_rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
                        radius=h / 2, fill=(*col, 255))
    label = "SS" if grade in ("SS", "X", "XH") else grade
    font = _font_body(max(int(h * (0.72 if len(label) == 1 else 0.56)), 6))
    try:
        x0, y0, x1, y1 = font.getbbox(label)
    except AttributeError:
        x1, y1 = font.getsize(label); x0 = y0 = 0    # type: ignore
    d.text((cx - (x1 - x0) / 2 - x0, cy - (y1 - y0) / 2 - y0), label,
           font=font, fill=(255, 255, 255, 245))


_ARC_GRAD_CACHE: dict = {}

# ---------------------------------------------------------------------------
# SHARED ARC CACHE + PREBAKE
#
# `_draw_acc_arc` keys its per-instance cache on `round(prog, 3)`, but `prog`
# advances every frame during the 1150 ms sweep, so the bucket changes every
# frame and the bake runs EVERY frame. Measured: 2.841 ms of a 4.46 ms outro
# frame -- 64% of the featured panel, the single biggest remaining cost.
#
# `bake_accuracy_arc` is a pure function of (px, bucket) -- deliberately, so the
# parallel outro could not diverge -- which makes it pre-bakeable with
# bit-identical results. The sweep is `ease_out_cubic` over a known window, so
# every bucket it will ever need is enumerable BEFORE gameplay starts, and the
# render and writer threads sit idle 88% / 75% of the wall with nothing to do.
#
# A miss here is a performance miss, never a correctness one: the caller falls
# back to baking its own, and gets the same pixels.
_ARC_SHARED: dict = {}
_ARC_LOCK = None


def _arc_lock():
    global _ARC_LOCK
    if _ARC_LOCK is None:
        import threading
        _ARC_LOCK = threading.Lock()
    return _ARC_LOCK


def _sweep_ages(frame_ms: float):
    """Every age the render loop can feed during the sweep.

    The loop computes `t = int(base + n*frame_ms)` and then `age = t - start`,
    so the ages are INTEGERS spaced 16 or 17 ms apart depending on where `base`
    falls -- NOT exact multiples of frame_ms. Enumerating only the exact
    multiples left 17 arc buckets uncovered and 0.576 ms/frame of bakes behind.
    Unioning every integer phase covers them all; enumerating every integer ms
    instead would be 10x the work for the same coverage.
    """
    end = SWEEP_DELAY_MS + SWEEP_MS + frame_ms * 2.0
    n_end = int(end / frame_ms) + 3
    ages = {i * frame_ms for i in range(n_end)}
    for ph in range(max(int(round(frame_ms)), 1) + 1):
        ages |= {float(int(ph + i * frame_ms) - ph) for i in range(n_end)}
    return ages


def _sweep_at(age: float) -> float:
    return ease_out_cubic((age - SWEEP_DELAY_MS) / SWEEP_MS) \
        if age > SWEEP_DELAY_MS else 0.0


def prebake_arcs(px: int, target_arc: float, frame_ms: float = 1000.0 / 60.0,
                 report=None) -> int:
    """Bake every arc bucket the sweep can produce. Safe to call from any
    thread at any time; returns how many it baked."""
    px = max(int(px), 64)
    buckets = {round(target_arc * _sweep_at(a), 3)
               for a in _sweep_ages(frame_ms)}
    buckets.add(round(target_arc, 3))          # the settled value
    n = 0
    for b in sorted(buckets):
        key = (px, b)
        if key in _ARC_SHARED:
            continue
        img = bake_accuracy_arc(px, b)
        with _arc_lock():
            _ARC_SHARED.setdefault(key, img)
        n += 1
    if report is not None:
        report(n, len(buckets))
    return n


# Same story as the arcs: `_score_text` rebakes the rolled score with
# bake_text() -- a supersampled render plus a LANCZOS downscale -- every time
# the value changes, which during the sweep is every frame. Measured 1.17 ms of
# a 4.07 ms panel. The value is a pure function of the age, so it is
# enumerable ahead of time exactly like the arc buckets.
_SCORE_SHARED: dict = {}


def prebake_scores(px: int, score: int, frame_ms: float = 1000.0 / 60.0,
                   report=None) -> int:
    px = max(int(px), 6)
    vals = {int(round(int(score) * _sweep_at(a)))
            for a in _sweep_ages(frame_ms)}
    vals.add(int(score))                       # the settled value
    n = 0
    for v in sorted(vals):
        key = (px, v)
        if key in _SCORE_SHARED:
            continue
        img = bake_text(f"{v:,}", px, (1, 1, 1), loader=_font_score)
        with _arc_lock():
            _SCORE_SHARED.setdefault(key, img)
        n += 1
    if report is not None:
        report(n, len(vals))
    return n


def prebake_for(meta, height: int, fps: float, report=None) -> int:
    """Work out px and target_arc the same way CatchLazerResults does, then
    prebake. If this ever drifts from __init__, the keys simply will not match
    and every frame falls back to baking -- slower, never wrong."""
    k = float(height) / UH
    px = int(ACC_DISP * k)
    caught = meta.count_300 + meta.count_100 + meta.count_50
    total = caught + meta.count_katu + meta.count_miss
    acc = (caught / total) if total else 1.0
    ta = target_arc_value(acc, catch_grade(acc))
    fm = 1000.0 / max(float(fps), 1.0)
    n = prebake_arcs(px, ta, fm, report)
    try:
        n += prebake_scores(int(64 * k), int(meta.score), fm)
    except Exception:      # noqa: BLE001 - optional, falls back to per-frame
        pass
    return n



def _arc_gradient_rgb(S: int):
    """Cached S×S vertical cyan→green gradient (std port)."""
    g = _ARC_GRAD_CACHE.get(S)
    if g is None:
        import numpy as np
        ys = np.linspace(0.0, 1.0, S).reshape(S, 1)
        top = np.array(ARC_GRAD_TOP)
        bot = np.array(ARC_GRAD_BOT)
        col = top + (bot - top) * ys
        rgb = np.repeat(col[:, None, :], S, axis=1)
        g = np.clip(np.round(rgb * 255), 0, 255).astype("u1")
        _ARC_GRAD_CACHE[S] = g
    return g


def bake_accuracy_arc(px: int, progress_acc: float,
                      ss: int = SHAPE_SS) -> Image.Image:
    """The achieved-accuracy arc (0 → progress_acc) — lazer's "Accuracy
    circle": the FIXED vertical cyan→green gradient with a light tip dash
    (std port). Re-baked per progress bucket while sweeping."""
    import numpy as np
    S_target = max(int(px), 64)
    ss = max(int(ss), 1)
    S = S_target * ss
    cx = cy = S / 2.0
    R = ACC_ARC_R * S
    W = max(int(round(ACC_ARC_W * S)), 2)
    mask = Image.new("L", (S, S), 0)
    if progress_acc > 0.0005:
        ImageDraw.Draw(mask).arc(
            [cx - R, cy - R, cx + R, cy + R], acc_to_angle_deg(0.0),
            acc_to_angle_deg(progress_acc), fill=255, width=W)
    rgb = _arc_gradient_rgb(S)
    out = np.dstack([rgb, np.asarray(mask, dtype="u1")]).copy()
    img = Image.fromarray(out, "RGBA")
    if progress_acc > 0.0005:
        d = ImageDraw.Draw(img)
        ang = math.radians(acc_to_angle_deg(progress_acc))
        r0 = R - W / 2.0 - 1
        r1 = R + W / 2.0 + 1
        d.line([(cx + r0 * math.cos(ang), cy + r0 * math.sin(ang)),
                (cx + r1 * math.cos(ang), cy + r1 * math.sin(ang))],
               fill=(255, 255, 255, 235), width=max(int(W * 0.16), 2))
    if ss != 1:
        img = img.resize((S_target, S_target), Image.LANCZOS)
    return img


def bake_star_pill(stars: float | None, k: float) -> Image.Image:
    """The StarRatingDisplay pill (std's _bake_star_pill): rounded
    ForStarDifficulty-coloured pill + procedural star + the star number,
    flipping to dark text on light backgrounds. Supersampled."""
    txt = f"{stars:.2f}" if stars is not None else "--"
    bg = for_star_difficulty(stars if stars is not None else 0.0)
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    fg = (0.06, 0.06, 0.09) if lum > 0.6 else (1.0, 1.0, 1.0)
    fpx = max(int(22 * k), 8)
    font = _font_body(fpx)
    try:
        x0, y0, x1, y1 = font.getbbox(txt)
    except AttributeError:
        x1, y1 = font.getsize(txt); x0 = y0 = 0     # type: ignore
    tw, th = x1 - x0, y1 - y0
    star_sz = int(fpx * 0.92)
    padx = int(fpx * 0.5)
    pady = int(fpx * 0.32)
    gap = int(fpx * 0.26)
    H = max(th, star_sz) + 2 * pady
    W = padx + star_sz + gap + tw + padx
    ss = TEXT_SS
    fpxb = fpx * ss
    fb = _font_body(fpxb)
    try:
        bx0, by0, bx1, by1 = fb.getbbox(txt)
    except AttributeError:
        bx1, by1 = fb.getsize(txt); bx0 = by0 = 0   # type: ignore
    twb, thb = bx1 - bx0, by1 - by0
    star_szb = star_sz * ss
    padxb, padyb, gapb = padx * ss, pady * ss, gap * ss
    Hb = max(thb, star_szb) + 2 * padyb
    Wb = padxb + star_szb + gapb + twb + padxb
    img = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    dd = ImageDraw.Draw(img)
    bgc = tuple(int(round(c * 255)) for c in bg)
    dd.rounded_rectangle([0, 0, Wb - 1, Hb - 1], radius=Hb // 2,
                         fill=(*bgc, 255))
    img.alpha_composite(bake_star(star_szb, fg),
                        (padxb, (Hb - star_szb) // 2))
    fgc = tuple(int(round(c * 255)) for c in fg)
    dd.text((padxb + star_szb + gapb - bx0, (Hb - thb) // 2 - by0), txt,
            font=fb, fill=(*fgc, 255))
    return img.resize((W, H), Image.LANCZOS)


def bake_mod_pill(text: str, color, k: float) -> Image.Image:
    """One played-mod badge (std's _bake_mod_pill): rounded category-coloured
    pill with the acronym in white, fixed height, supersampled."""
    H = max(int(round(MOD_PILL_VH * k)), 12)
    fpx = max(int(round(MOD_PILL_TEXT_VPX * k)), 8)
    padx = int(round(fpx * 0.62))
    ss = TEXT_SS
    Hb = H * ss
    fb = _font_body(fpx * ss)
    try:
        bx0, by0, bx1, by1 = fb.getbbox(text)
    except AttributeError:
        bx1, by1 = fb.getsize(text); bx0 = by0 = 0    # type: ignore
    twb, thb = bx1 - bx0, by1 - by0
    padxb = padx * ss
    Wb = twb + 2 * padxb
    W = max(int(round(Wb / ss)), 8)
    img = Image.new("RGBA", (Wb, Hb), (0, 0, 0, 0))
    dd = ImageDraw.Draw(img)
    bgc = tuple(int(round(c * 255)) for c in color)
    dd.rounded_rectangle([0, 0, Wb - 1, Hb - 1], radius=Hb // 2,
                         fill=(*bgc, MOD_PILL_ALPHA))
    ty = (Hb - thb) // 2 - by0
    dd.text((padxb - bx0, ty), text, font=fb, fill=(255, 255, 255, 255))
    return img.resize((W, H), Image.LANCZOS)


def _bake_area_chart(values: list, w: int, h: int,
                     tick_fracs: list[float] | None = None) -> Image.Image:
    """A filled area chart of `values` (left→right) for the stage-2 COMBO /
    DIFFICULTY panel: the area under the curve wears the results screen's
    cyan→green accent gradient (the accuracy arc's), a brighter line rides
    the top, a dim baseline sits underneath, and optional red ticks mark
    combo breaks at the given x fractions. Supersampled 2× then LANCZOS-
    downscaled. Returns a PIL RGBA image (w×h)."""
    w = max(int(w), 32)
    h = max(int(h), 24)
    ss = 2
    W, H = w * ss, h * ss
    vals = [max(float(v), 0.0) for v in values] or [0.0]
    # peak-preserving resample so short combo spikes / strain peaks survive
    cols = min(len(vals), max(w // 3, 60))
    step = len(vals) / cols
    res = [max(vals[int(i * step): max(int((i + 1) * step),
                                       int(i * step) + 1)] or [0.0])
           for i in range(cols)]
    peak = max(res) or 1.0
    inset = max(int(H * 0.04), 2)              # headroom above the peak
    base_y = H - max(int(2 * ss), 2)           # baseline (bottom)
    span_h = base_y - inset
    pts = [(i * (W - 1) / max(cols - 1, 1),
            base_y - span_h * (v / peak)) for i, v in enumerate(res)]
    # gradient-filled area under the curve
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(
        [(0, base_y)] + pts + [(W - 1, base_y)], fill=255)
    import numpy as np
    ys = np.linspace(0.0, 1.0, H).reshape(H, 1, 1)
    grad = (np.array(ARC_GRAD_TOP) + (np.array(ARC_GRAD_BOT)
                                      - np.array(ARC_GRAD_TOP)) * ys)
    rgb = np.clip(np.round(np.repeat(grad, W, axis=1) * 255), 0,
                  255).astype("u1")
    alpha = (np.asarray(mask, dtype="f4") * (0.62 / 255.0) * 255) \
        .astype("u1")
    img = Image.fromarray(np.dstack([rgb, alpha]), "RGBA")
    d = ImageDraw.Draw(img)
    # dim baseline + the brighter curve line on top
    d.line([(0, base_y), (W - 1, base_y)], fill=(255, 255, 255, 90),
           width=max(ss, 1))
    d.line(pts, fill=(232, 255, 245, 235), width=2 * ss, joint="curve")
    # red combo-break ticks
    for f in tick_fracs or []:
        x = _clamp01(f) * (W - 1)
        d.line([(x, inset), (x, base_y)], fill=(255, 90, 90, 150),
               width=max(ss, 1))
    return img.resize((w, h), Image.LANCZOS)


# --- data helpers --------------------------------------------------------------------

def _compute_stars_pp(osu_path, mods: int, meta):
    """(stars, pp, max_pp) via rosu-pp with the Catch-ruleset conversion (the
    same path scene.compute_pp_curve uses). `max_pp` = the perfect-play (SS)
    pp for the stage-2 PERFORMANCE footer. Fail-soft → (None, None, None)."""
    if osu_path is None:
        return None, None, None
    try:
        import rosu_pp_py as rosu
        rbm = rosu.Beatmap(path=str(osu_path))
        try:
            if rbm.mode != rosu.GameMode.Catch:
                rbm.convert(rosu.GameMode.Catch, int(mods))
        except Exception:  # noqa: BLE001 — conversion refusal → raw map
            pass
        stars = None
        try:
            stars = float(rosu.Difficulty(mods=int(mods)).calculate(rbm).stars)
        except Exception:  # noqa: BLE001 — stars are optional
            stars = None
        pp = None
        try:
            pp = float(rosu.Performance(
                mods=int(mods), n300=meta.count_300, n100=meta.count_100,
                n50=meta.count_50, n_katu=meta.count_katu,
                misses=meta.count_miss, combo=meta.max_combo,
            ).calculate(rbm).pp)
        except Exception:  # noqa: BLE001 — pp is optional
            pp = None
        max_pp = None
        try:
            # no hitresults → rosu assumes a perfect play (the SS ceiling)
            max_pp = float(rosu.Performance(mods=int(mods)).calculate(rbm).pp)
        except Exception:  # noqa: BLE001 — the ceiling is optional
            max_pp = None
        return stars, pp, max_pp
    except Exception:  # noqa: BLE001 — no rosu / unreadable map → no row values
        return None, None, None


def _compute_strains(osu_path, mods: int) -> list[float]:
    """The rosu-pp MOVEMENT strain curve (catch's single skill) — map
    difficulty over time, the stage-2 COMBO panel's fallback chart when the
    sim's combo series isn't available. Fail-soft → []."""
    if osu_path is None:
        return []
    try:
        import rosu_pp_py as rosu
        rbm = rosu.Beatmap(path=str(osu_path))
        try:
            if rbm.mode != rosu.GameMode.Catch:
                rbm.convert(rosu.GameMode.Catch, int(mods))
        except Exception:  # noqa: BLE001 — conversion refusal → raw map
            pass
        vals = list(rosu.Difficulty(mods=int(mods)).strains(rbm).movement
                    or [])
        return [float(v) for v in vals if v == v]      # drop NaNs
    except Exception:  # noqa: BLE001 — no rosu / no strains → no chart
        return []


# The FEATURED (current) player's REAL osu! avatar PNG bytes, set by the CLI
# (--featured-avatar-png; osu! user → avatar_url → PNG) via
# set_featured_avatar_png() before the render loop. The featured centre card
# uses these when present, else the procedural username chip. Replaces the old
# render-DB Discord-id lookup, which could resolve a player's name to the SITE
# OWNER's Discord pfp on the featured card (bug 2026-07-11). The flank cards
# keep their own Discord avatars (lb_cards -- untouched).
_FEATURED_AVATAR_PNG_BYTES: bytes | None = None


def set_featured_avatar_png(path) -> None:
    """Load the featured player's osu! avatar PNG bytes from `path` (called by
    the CLI once, before rendering). Fail-soft: a falsy/missing/unreadable path
    leaves the featured card on the procedural chip."""
    global _FEATURED_AVATAR_PNG_BYTES
    try:
        from pathlib import Path as _P
        _FEATURED_AVATAR_PNG_BYTES = _P(path).read_bytes() if path else None
    except OSError:
        _FEATURED_AVATAR_PNG_BYTES = None


def _featured_avatar_bytes(player_name: str, beatmap_md5: str):
    """PNG bytes of the FEATURED player's REAL osu! avatar (set by the CLI via
    set_featured_avatar_png / --featured-avatar-png), or None → the procedural
    username chip. The old render-DB Discord path was REMOVED: a player whose
    name maps (stale/colliding) to a linked Discord account could resolve to
    the SITE OWNER's pfp on the featured card (bug 2026-07-11). The flank cards
    keep their own Discord avatars (lb_cards -- untouched)."""
    return _FEATURED_AVATAR_PNG_BYTES


class DrawRecorder:
    """Stands in for the PIL `base` image and records what WOULD be composited.

    Every element of the results screen is pasted through `_paste` (24 sites)
    or `_blit` (3), so swapping `base` for this captures the complete draw list
    -- geometry from the SAME layout code the CPU path uses, which is the whole
    point: a re-derived layout would drift. The exception is `_draw_bar_rows`,
    which uses ImageDraw directly; that is stage-2 only and unsupported here.
    """

    __slots__ = ("ops", "size", "label", "names")

    is_recorder = True

    def __init__(self, size, label=False):
        self.ops = []            # (img, x0, y0, alpha)
        self.size = size
        self.label = label       # record call sites too (debug dump only)
        self.names = []

    def alpha_composite(self, img, pos):          # pragma: no cover - guard
        raise AssertionError("recorder: unexpected direct alpha_composite")


def _paste(base, img: Image.Image, cx: float, cy: float,
           alpha: float) -> None:
    """Composite `img` centred at (cx, cy) at `alpha` (lb_cards._paste)."""
    if alpha <= 0.003 or img is None:
        return
    if getattr(base, "is_recorder", False):
        if base.label:
            import sys as _ls
            f = _ls._getframe(1)
            base.names.append(f"{f.f_code.co_name}:{f.f_lineno}")
        # record the UNSCALED tile plus the alpha -- the GPU applies alpha
        # per-instance, so pre-multiplying it into a copy here would be waste
        base.ops.append((img, int(cx - img.width / 2.0),
                         int(cy - img.height / 2.0), alpha))
        return
    if alpha < 0.997:
        a = img.getchannel("A").point(lambda v: int(v * alpha))
        img = img.copy()
        img.putalpha(a)
    base.alpha_composite(img, (int(cx - img.width / 2.0),
                               int(cy - img.height / 2.0)))


# --- the screen ----------------------------------------------------------------------

class CatchLazerResults:
    """Bakes the lazer ranking screen once, animates the two-stage reveal per
    frame on catch's CPU compositor, then caches the settled frame."""

    def __init__(self, resolution, meta, bm, board=None, osu_path=None,
                 sim=None, pp_override=None, sr_override=None):
        self.W, self.H = int(resolution[0]), int(resolution[1])
        self.k = self.H / UH
        self.meta = meta
        self.bm = bm
        self.board = board                 # lb_cards.BakedBoard | None
        self._settled = None               # cached final RGB frame
        # pose-signature frame memo (PERF, output-identical): every animated
        # parameter of the screen is a CLAMPED pure function of (op, age_ms),
        # so once they all saturate — or hold between animation phases — the
        # rendered frame is a pure function of the signature. Reusing the
        # previous frame whenever the signature is unchanged skips the whole
        # composite for the static tail of the outro (~1/5 of it) without
        # touching a single animated frame.
        self._pose_key = None
        self._pose_frame = None
        self._base_panel: dict = {}      # black wash + panel bg, reused
        self._arc_img = None
        self._arc_bucket = -1.0
        self._score_img = None
        self._score_val = -1
        self._osu_path = osu_path

        # --- results data (the replay's authoritative counts) ---------------
        caught = meta.count_300 + meta.count_100 + meta.count_50
        total = caught + meta.count_katu + meta.count_miss
        self.acc_frac = (caught / total) if total else 1.0
        self.acc_pct = 100.0 * self.acc_frac
        self.grade = catch_grade(self.acc_frac)
        self.grade_rgb = FOR_RANK.get(self.grade, (0.8, 0.8, 0.85))
        self.target_arc = target_arc_value(self.acc_frac, self.grade)
        # MISS = fruit + large-droplet misses ONLY (count_miss). Tiny-droplet
        # misses (count_katu) are NOT misses in either client — folding them
        # in showed 26 when the game showed 10 (owner repro 2026-07-23). The
        # missed tinies stay visible via the DROPLET caught/total cell below.
        self.miss_display = meta.count_miss
        self.stars, self.pp, self.max_pp = _compute_stars_pp(
            osu_path, meta.mods, meta)
        # --pp: pin the results-card PP to the EXACT official pp passed by
        # the service (RenderConfig.pp_override), overriding the rosu
        # estimate. Every results PP consumer reads self.pp -- the PP grid
        # cell, the stage-2 PP perf bar (against the rosu SS ceiling
        # self.max_pp, kept), and the "Achieved Npp" footer -- so all three
        # show the official value. max_pp/stars stay rosu. None -> unchanged
        # rosu pp. Mirrors the taiko _final_pp override.
        if pp_override is not None:
            self.pp = float(pp_override)
        # --sr: pin the results-card star-rating pill to the EXACT official SR
        # passed by the service (RenderConfig.sr_override), overriding the rosu
        # estimate. self.stars feeds bake_star_pill (the pill number + colour).
        # max_pp/pp stay as computed. None -> unchanged rosu SR. Mirrors the
        # pp_override above.
        if sr_override is not None:
            self.stars = float(sr_override)

        # --- stage-2 source data (real catch data; never faked) -------------
        # combo-over-time from the sim's per-object checkpoints + the number
        # of hyperdashes the play actually used (the sim's hyper windows).
        # Fail-soft: no sim (legacy caller) → empty series → the COMBO panel
        # falls back to the rosu movement-strain curve at bake time.
        self._combo_series: list[tuple[int, int]] = []
        self._hyper_used: int | None = None
        try:
            if sim is not None:
                cps = getattr(sim, "_checkpoints", None) or []
                self._combo_series = [(int(c.time), int(c.combo))
                                      for c in cps]
                hw = getattr(sim, "_hyper_windows", None)
                if hw is not None:
                    self._hyper_used = len(hw)
        except Exception:  # noqa: BLE001 — stage-2 data never breaks a bake
            self._combo_series = []
            self._hyper_used = None
        self._bake_static()

    # -- baking ------------------------------------------------------------------

    def _bake_static(self) -> None:
        k = self.k
        m = self.meta
        bm = self.bm
        self.cx_px = self.W / 2.0
        self.panel_cy_v = UH / 2.0
        self.panel_cy_px = self.panel_cy_v * k
        self.panel_top_px = (self.panel_cy_v - PANEL_H / 2.0) * k
        # main panel bg (std's colours/radius/border)
        self.panel_img = bake_round_panel(
            int(PANEL_W * k), int(PANEL_H * k), int(26 * k),
            (0.12, 0.13, 0.17), (0.05, 0.05, 0.07), 0.93,
            border=(0.3, 0.33, 0.4))
        # avatar + header rows (auto-fit, std budgets)
        self.avatar_img = bake_avatar(
            int(52 * k), m.player_name or "?",
            _featured_avatar_bytes(m.player_name or "",
                                   getattr(m, "beatmap_md5", "") or ""))
        content_w = PANEL_W - 48.0
        self.name_img = fit_text(m.player_name or "Player", 30, (1, 1, 1),
                                 content_w - 64.0, k)
        self.title_img = fit_text(bm.title or "", 34, (0.95, 0.96, 1.0),
                                  content_w, k)
        self.artist_img = fit_text(bm.artist or "", 24, (0.72, 0.75, 0.85),
                                   content_w, k)
        # accuracy circle base + centre grade letter (white + ForRank glow)
        self.acc_base_img = bake_accuracy_base(int(ACC_DISP * k))
        self.grade_img = bake_grade_letter(self.grade, int(150 * k),
                                           (0.99, 0.99, 1.0), self.grade_rgb)
        # score baked lazily (rolls); seed with 0
        self._score_text(0)
        # star pill + played-mods badges + diff/creator rows
        self.star_pill = bake_star_pill(self.stars, k)
        acr = [a for a in mods_string(int(m.mods or 0)).split(",") if a]
        self.mod_pills = [bake_mod_pill(a, mod_pill_color(a), k) for a in acr]
        self.diff_img = bake_text(_clip(bm.version, 28), int(26 * k),
                                  (0.9, 0.92, 1.0))
        creator = getattr(bm, "creator", "") or ""
        self.creator_img = bake_text(
            f"mapped by {_clip(creator, 22)}" if creator else "",
            int(22 * k), (0.65, 0.68, 0.78))
        # The star/diff/creator row is CENTRED in _draw_star_row — without a
        # PIXEL budget a long diff name + "mapped by …" made the row wider
        # than the panel, running off both card edges and pushing the star
        # pill off the LEFT edge (char-count _clip alone doesn't bound px).
        # Only refit when the natural bake overflows, so every fitting row
        # stays bit-identical to the old output.
        _row_gap = 14.0 * k
        _row_avail = max(content_w * k - self.star_pill.width - 2.0 * _row_gap,
                        1.0)
        if self.diff_img.width + self.creator_img.width > _row_avail:
            _d_budget = min(float(self.diff_img.width),
                            max(_row_avail - self.creator_img.width,
                                _row_avail * 0.5))
            self.diff_img = fit_text(bm.version or "", 26.0, (0.9, 0.92, 1.0),
                                     _d_budget / k, k)
            _c_budget = max(_row_avail - self.diff_img.width, 1.0)
            self.creator_img = fit_text(
                f"mapped by {creator}" if creator else "", 22.0,
                (0.65, 0.68, 0.78), _c_budget / k, k)
        # stats grids — catch judgments (no slider tick/end rows: catch has
        # no such concept). Judgment colours = the flank cards' palette.
        pp_txt = (f"{self.pp:.0f}" if self.pp is not None else "--")
        grid_a = [
            ("ACCURACY", f"{self.acc_pct:.2f}%", (0.95, 0.96, 1.0)),
            ("MAX COMBO", f"{m.max_combo}x", (0.95, 0.96, 1.0)),
            ("PP", pp_txt, (0.6, 0.86, 1.0)),
        ]
        grid_b = [
            ("FRUIT", str(m.count_300), CATCH_RESULT_COLORS["FRUIT"]),
            ("DROP", str(m.count_100), CATCH_RESULT_COLORS["DROP"]),
            ("DROPLET",
             (f"{m.count_50}/{m.count_50 + m.count_katu}"
              if m.count_katu else str(m.count_50)),
             CATCH_RESULT_COLORS["DROPLET"]),
            ("MISS", str(self.miss_display), CATCH_RESULT_COLORS["MISS"]),
        ]
        self.grid_a = [self._grid_cell(lbl, val, col) for lbl, val, col in grid_a]
        self.grid_b = [self._grid_cell(lbl, val, col) for lbl, val, col in grid_b]
        ts = getattr(m, "timestamp", None)
        if not isinstance(ts, datetime):
            ts = datetime.now()
        self.date_img = bake_text(f"Played on {ts.strftime('%d %b %Y %H:%M')}",
                                  int(20 * k), (0.6, 0.63, 0.72))
        # stage 2: the featured panel's slide-left target + the stats panels
        # that unfold from the right (std geometry, catch data). Fail-soft:
        # any bake problem disables stage 2 LOUDLY and the screen holds
        # centred exactly as before the port.
        self._stage2 = False
        try:
            self._bake_stage2()
        except Exception as e:  # noqa: BLE001 — stage 2 never breaks results
            import sys
            import traceback
            print("[catch-renderer] !!! RESULTS STAGE-2 BAKE FAILED — "
                  f"holding the stage-1 screen: {e}", file=sys.stderr)
            traceback.print_exc()
            self._stage2 = False

    def _grid_cell(self, label, value, color):
        return (bake_text(label, int(16 * self.k), (0.6, 0.63, 0.73)),
                bake_text(value, int(32 * self.k), color))

    def _score_text(self, value: int) -> None:
        px = int(64 * self.k)
        img = _SCORE_SHARED.get((px, value))
        if img is None:
            img = bake_text(f"{value:,}", px, (1, 1, 1), loader=_font_score)
        self._score_img = img
        self._score_val = value

    # -- stage-2 bake ------------------------------------------------------------

    def _bake_stage2(self) -> None:
        """Stage-2 geometry + content (std's exact layout, catch's real data).

        std geometry (render/lazer_results.py): the panel slides from the
        screen centre to left_cx = PANEL_W/2 + 70; the three stats panels
        live at STATS_X0 = left_cx + PANEL_W/2 + 40, filling the rest of the
        width minus a 64-virtual-px right margin, each (PANEL_H − 2·24)/3
        tall with 24 vpx gaps. Content is baked ONCE here; only the panel
        squash + alpha animate per frame."""
        k = self.k
        m = self.meta
        self.uw = self.W / k                       # virtual width
        self.center_cx_v = self.uw / 2.0
        self.left_cx_v = PANEL_W / 2.0 + 70.0
        self.STATS_X0 = self.left_cx_v + PANEL_W / 2.0 + 40.0
        self.STATS_W = self.uw - self.STATS_X0 - 64.0
        self.STATS_H = (PANEL_H - 2 * 24.0) / 3.0
        if self.STATS_W < 200.0:      # too narrow (portrait render) → stage 1 only
            return
        self.stats_panel_img = bake_round_panel(
            int(self.STATS_W * k), int(self.STATS_H * k), int(20 * k),
            (0.11, 0.12, 0.16), (0.05, 0.05, 0.07), 0.93,
            border=(0.28, 0.31, 0.38))
        title_col = (0.8, 0.83, 0.92)
        label_col = (0.85, 0.88, 0.95)
        foot_col = (0.62, 0.66, 0.76)

        # --- panel 1: PERFORMANCE (acc / combo / pp bars, all real) ---------
        self.perf_title = bake_text("PERFORMANCE", int(22 * k), title_col)
        self._perf_rows = []           # (label_img, pct_img, pct, colour)

        def _perf_row(label, pct, colour, fmt=".0f"):
            self._perf_rows.append((
                bake_text(label, int(20 * k), label_col),
                bake_text(f"{pct * 100:{fmt}}%", int(20 * k), (1, 1, 1)),
                _clamp01(pct), colour))

        # accuracy keeps 2 decimals — .0f would round 99.55% up to "100%"
        # right next to the main panel's exact figure
        _perf_row("Accuracy", self.acc_frac, CATCH_RESULT_COLORS["FRUIT"],
                  fmt=".2f")
        objs = getattr(self.bm, "objects", None) or []
        n_fruit = sum(1 for o in objs if o.kind.name == "FRUIT")
        n_drop = sum(1 for o in objs if o.kind.name == "DROPLET")
        n_banana = sum(1 for o in objs if o.kind.name == "BANANA")
        self._map_max_combo = n_fruit + n_drop     # tiny droplets don't combo
        if self._map_max_combo > 0:
            _perf_row("Combo", m.max_combo / self._map_max_combo,
                      CATCH_RESULT_COLORS["DROP"])
        if self.pp is not None and self.max_pp:
            _perf_row("PP", self.pp / self.max_pp, (0.6, 0.86, 1.0))
        if self.pp is not None and self.max_pp is not None:
            foot = (f"Achieved {self.pp:.0f}pp  /  "
                    f"Maximum {self.max_pp:.0f}pp")
        else:
            foot = "performance unavailable (no rosu)"
        self.perf_foot = bake_text(foot, int(18 * k), foot_col)

        # --- panel 2: COMBO progression (sim) / DIFFICULTY strain (rosu) ----
        pad = 26.0 * k
        th = self.perf_title.height
        chart_w = int(self.STATS_W * k - 2 * pad)
        chart_h = int(self.STATS_H * k - (22 * k + th) - 54 * k)
        series = self._combo_series
        self.combo_chart = None
        if len(series) >= 8:
            self.combo_title = bake_text("COMBO", int(22 * k), title_col)
            vals = [c for _t, c in series]
            # combo breaks: a reset to 0 after a positive run (tiny-droplet
            # misses don't reset, so every 0-drop here is a real break)
            breaks = [i for i in range(1, len(vals))
                      if vals[i] == 0 and vals[i - 1] > 0]
            tick_fr = [i / max(len(vals) - 1, 1) for i in breaks]
            self.combo_chart = _bake_area_chart(vals, chart_w, chart_h,
                                                tick_fracs=tick_fr)
            self.combo_foot = bake_text(
                f"peak {m.max_combo}x   ·   {len(breaks)} combo "
                f"break{'' if len(breaks) == 1 else 's'}",
                int(18 * k), foot_col)
        else:
            strains = _compute_strains(self._osu_path, int(m.mods or 0))
            if len(strains) >= 4:
                self.combo_title = bake_text("DIFFICULTY", int(22 * k),
                                             title_col)
                self.combo_chart = _bake_area_chart(strains, chart_w, chart_h)
                self.combo_foot = bake_text("movement strain over time",
                                            int(18 * k), foot_col)
            else:                       # honest: nothing graphable
                self.combo_title = bake_text("COMBO", int(22 * k), title_col)
                self.combo_foot = bake_text("timeline unavailable",
                                            int(18 * k), foot_col)

        # --- panel 3: CATCHES (caught/total per type + hyperdash usage) -----
        self.catch_title = bake_text("CATCHES", int(22 * k), title_col)
        self._catch_rows = []          # (label_img, count_img, frac, colour)
        for label, caught, total_n, ckey in (
                ("Fruits", m.count_300, n_fruit, "FRUIT"),
                ("Drops", m.count_100, n_drop, "DROP"),
                # tiny-droplet TOTAL is the .osr header's authoritative count
                # (caught count_50 + missed count_katu), NOT the parsed tiny
                # count: the geometric parse can generate more tiny droplets
                # than the real play had (bit-exact RNG spacing we can't
                # reproduce), which showed a bogus "724/778" on a genuine SS.
                ("Droplets", m.count_50, m.count_50 + m.count_katu, "DROPLET")):
            if total_n <= 0 and caught <= 0:
                continue
            frac = _clamp01(caught / total_n) if total_n > 0 else 1.0
            self._catch_rows.append((
                bake_text(label, int(20 * k), label_col),
                bake_text(f"{caught}/{total_n}" if total_n > 0
                          else str(caught), int(20 * k), (1, 1, 1)),
                frac, CATCH_RESULT_COLORS[ckey]))
        hyper_total = sum(1 for o in objs if getattr(o, "hyperdash", False))
        if hyper_total <= 0:
            foot = "no hyperdashes"
        elif self._hyper_used is not None:
            foot = (f"{min(self._hyper_used, hyper_total)}/{hyper_total} "
                    "hyperdashes used")
        else:
            foot = f"{hyper_total} hyperdashes in map"
        if n_banana > 0:
            foot += f"   ·   {n_banana} bananas"
        self.catch_foot = bake_text(foot, int(18 * k), foot_col)
        self._stage2 = True

    # -- per-frame draw ----------------------------------------------------------

    # R3D_RESULTS_PERF=1: where the outro's ms/frame actually goes.
    _RP = {"n": 0, "hit_settled": 0, "hit_pose": 0, "miss": 0, "base": 0.0,
           "board": 0.0, "stats": 0.0, "panel": 0.0, "out": 0.0, "total": 0.0,
           # panel sub-stages: the panel is ~58% of a rendered outro frame,
           # so knowing WHICH part of it costs the 7.81 ms is the whole point
           "p_bg": 0.0, "p_header": 0.0, "p_titles": 0.0, "p_accbase": 0.0,
           "p_arc": 0.0, "p_arcbake": 0.0, "p_grade": 0.0, "p_score": 0.0,
           "p_mods": 0.0, "p_star": 0.0, "p_grids": 0.0, "p_date": 0.0,
           "p_n": 0}

    @classmethod
    def _rp_report(cls):
        import sys
        d = cls._RP
        if not d["n"]:
            return
        m = max(d["miss"], 1)
        print("RESULTS-PERF frames=%d settled_hit=%d pose_hit=%d MISS=%d (%.0f%%)"
              % (d["n"], d["hit_settled"], d["hit_pose"], d["miss"],
                 100.0 * d["miss"] / d["n"]), file=sys.stderr)
        print("RESULTS-PERF per-MISS ms: base=%.2f board=%.2f stats=%.2f "
              "panel=%.2f asarray=%.2f | total/frame=%.2f"
              % (1e3 * d["base"] / m, 1e3 * d["board"] / m, 1e3 * d["stats"] / m,
                 1e3 * d["panel"] / m, 1e3 * d["out"] / m,
                 1e3 * d["total"] / max(d["n"], 1)), file=sys.stderr)
        pn = max(d["p_n"], 1)
        subs = [(k[2:], 1e3 * d[k] / pn) for k in
                ("p_bg", "p_header", "p_titles", "p_accbase", "p_arc",
                 "p_arcbake", "p_grade", "p_score", "p_mods", "p_star",
                 "p_grids", "p_date")]
        tot = sum(v for _, v in subs) or 1.0
        print("RESULTS-PERF panel breakdown (n=%d draws), ms/draw:" % d["p_n"],
              file=sys.stderr)
        for kk, vv in sorted(subs, key=lambda kv: -kv[1]):
            print("RESULTS-PERF   %-9s %6.3f ms  %4.1f%%" % (kk, vv,
                  100.0 * vv / tot), file=sys.stderr)

    def render_frame(self, rgb, opacity: float, age_ms: float | None,
                     capture: bool = False):
        """Composite the results screen over `rgb` (HxWx3 uint8) and return
        the finished RGB array. `opacity` = the render loop's results fade
        (drives the black wash + a global alpha); `age_ms` = ms since the
        results started (drives the ported stage-1 animation)."""
        import numpy as np
        import os as _os, time as _t
        _P = self._RP if _os.environ.get("R3D_RESULTS_PERF") else None
        _tA = _t.perf_counter() if _P else 0.0
        if _P: _P["n"] += 1
        op = _clamp01(opacity)
        if age_ms is None:                 # back-compat: no timeline → settled
            age_ms = SETTLE_MS if op >= 0.999 else op * FADE_MS
        settled = op >= 0.999 and age_ms >= SETTLE_MS
        if capture:
            return self._capture_ops(rgb, op, age_ms)
        if settled and self._settled is not None:
            if _P: _P["hit_settled"] += 1; _P["total"] += _t.perf_counter() - _tA
            return self._settled
        sig = self._pose_signature(op, age_ms)
        if sig is not None and sig == self._pose_key \
                and self._pose_frame is not None:
            if _P: _P["hit_pose"] += 1; _P["total"] += _t.perf_counter() - _tA
            return self._pose_frame
        if _P: _P["miss"] += 1
        if (_os.environ.get("R3D_OUTRO_ELEMENTS")
                and not getattr(CatchLazerResults, "_dumped", False)
                and self.gpu_supported(op, age_ms)):
            self._capture_ops(rgb, op, age_ms)      # dumps, then discarded
        _TR = None
        if _os.environ.get("R3D_RESULTS_TRACE"):
            _want = int(_os.environ["R3D_RESULTS_TRACE"])
            if _P is not None and _P["miss"] == _want:
                _TR = [("enter", _tA)]

        # Animation params hoisted ABOVE the base build: the black base and the
        # panel background are IDENTICAL on every frame where the wash is
        # opaque, the panel alpha has saturated, and nothing is drawn between
        # them -- so that pair can be built once and copied. Measured together
        # at 1.24 (base) + 2.03 (bg) = 3.27 ms per outro frame.
        _fade = _clamp01(age_ms / FADE_MS) * op
        _stage2 = getattr(self, "_stage2", False)
        _open_p = ease_out_quint((age_ms - STAGE1_MS) / OPEN_MS) \
            if (_stage2 and age_ms > STAGE1_MS) else 0.0
        _panel_cx = _lerp(self.center_cx_v, self.left_cx_v, _open_p) \
            * self.k if _stage2 else self.cx_px
        _stage1_a = _fade * (1.0 - _clamp01((age_ms - STAGE1_MS)
                                            / (OPEN_MS * 0.6))) \
            if _stage2 else _fade
        # Eligible only when NOTHING is composited between the base and the
        # panel background -- the board and the stats panels both draw first,
        # so their presence disqualifies the shortcut.
        _bp_key = None
        if (op >= 1.0 and _fade >= 0.997
                and not (self.board is not None and _stage1_a > 0.003)
                and not (_stage2 and age_ms > STAGE1_MS)):
            _bp_key = int(round(_panel_cx))
        _skip_bg = False
        if _bp_key is not None:
            _cached = self._base_panel.get(_bp_key)
            if _cached is None:
                _cached = Image.new("RGBA", (rgb.shape[1], rgb.shape[0]),
                                    (0, 0, 0, 255))
                _paste(_cached, self.panel_img, _panel_cx,
                       self.panel_cy_px, 1.0)
                if len(self._base_panel) < 4:      # bounded; cx is constant
                    self._base_panel[_bp_key] = _cached
            base = _cached.copy()
            _skip_bg = True

        # BLACK background: std's scene has faded to black by results_start
        # and the screen dims what's left — the results sit on clean black.
        # PERF: once the wash is fully opaque (op == 1.0 after the 400 ms
        # fade — most outro frames) compositing (0,0,0,255) over ANY frame
        # is exactly opaque black, so skip the two full-frame ops and start
        # from black directly (bit-identical: src a=255 ⇒ out = src).
        if _skip_bg:
            pass                                   # base already built above
        elif op >= 1.0:
            base = Image.new("RGBA", (rgb.shape[1], rgb.shape[0]),
                             (0, 0, 0, 255))
        else:
            # FADE-IN FRAMES. The old path was fromarray -> putalpha(255) ->
            # Image.new(black) -> alpha_composite: FOUR full-frame operations
            # over 8.3 MB, measured at 6.305 ms on a single frame -- the most
            # expensive stage in the whole outro.
            #
            # Compositing opaque black at alpha A over an opaque base is, in
            # PIL's own integer arithmetic, exactly `(f*(255-A) + 127) // 255`
            # per channel (verified against alpha_composite across A). That is
            # a scalar multiply, so one point() LUT does it in a single pass --
            # and point() is the fast path here (it beat numpy LUTs 9x when we
            # measured it on the death grade). The 4th LUT band pins alpha to
            # 255, which is what putalpha(255) was for.
            #
            # NOT prebakeable, unlike the arcs: op rises monotonically so every
            # value occurs exactly once, and it depends on the frozen final
            # gameplay frame, which does not exist until gameplay has ended.
            _A = int(op * 255)
            _lut = [(v * (255 - _A) + 127) // 255 for v in range(256)]
            if rgb.ndim == 3 and rgb.shape[2] == 4:
                base = Image.fromarray(rgb, "RGBA").point(
                    _lut * 3 + [255] * 256)
            else:
                base = Image.fromarray(rgb, "RGB").point(_lut * 3) \
                    .convert("RGBA")

        if _P: _P["base"] += _t.perf_counter() - _tA
        if _TR is not None: _TR.append(("base", _t.perf_counter()))
        fade = _fade
        if fade > 0.003:
            stage2, open_p, panel_cx, stage1_a = (
                _stage2, _open_p, _panel_cx, _stage1_a)
            _t0 = _t.perf_counter() if _P else 0.0
            if self.board is not None and stage1_a > 0.003:
                self._draw_board(base, age_ms, stage1_a,
                                 dx=panel_cx - self.cx_px)
            if _P: _t1 = _t.perf_counter(); _P["board"] += _t1 - _t0
            if _TR is not None: _TR.append(("board", _t.perf_counter()))
            if stage2 and age_ms > STAGE1_MS:
                self._draw_stats_panels(base, age_ms, op)
            if _P: _t2 = _t.perf_counter(); _P["stats"] += _t2 - _t1
            if _TR is not None: _TR.append(("stats", _t.perf_counter()))
            self._draw_panel(base, age_ms, fade, panel_cx, skip_bg=_skip_bg)
            if _P: _P["panel"] += _t.perf_counter() - _t2
            if _TR is not None: _TR.append(("panel", _t.perf_counter()))

        # RGBA pipeline: keep 4 channels when the input frame was 4ch — the
        # ffmpeg pipe is rgba now. base's alpha is opaque throughout (built
        # from putalpha(255) / (0,0,0,255)), the encoder ignores it anyway;
        # RGB values are identical to the old convert("RGB") return.
        _tC = _t.perf_counter() if _P else 0.0
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            out = np.asarray(base)
        else:
            out = np.asarray(base.convert("RGB"))
        if _P: _P["out"] += _t.perf_counter() - _tC
        if settled:
            self._settled = out
        if _TR is not None:
            _TR.append(("asarray/out", _t.perf_counter()))
            import sys as _trs
            t0 = _TR[0][1]
            print(f"OUTRO-FRAME TRACE  (miss #{_P['miss']}, age_ms={age_ms:.0f}, "
                  f"op={op:.3f}, stage2={getattr(self, '_stage2', False)}, "
                  f"reused_base={_skip_bg})", file=_trs.stderr)
            prev = t0
            for nm, tt in _TR[1:]:
                print(f"OUTRO-FRAME   {nm:12s} +{1e3*(tt-t0):6.3f} ms  "
                      f"(stage {1e3*(tt-prev):6.3f} ms)", file=_trs.stderr)
                prev = tt
            print(f"OUTRO-FRAME   TOTAL        {1e3*(prev-t0):6.3f} ms",
                  file=_trs.stderr)
        if (_os.environ.get("R3D_CAPTURE_VERIFY") and _P is not None
                and _P["miss"] == int(_os.environ["R3D_CAPTURE_VERIFY"])):
            self._verify_capture(rgb, op, age_ms, out)
        if sig is not None:
            self._pose_key, self._pose_frame = sig, out
        if _P: _P["total"] += _t.perf_counter() - _tA
        return out

    # ---- GPU capture ------------------------------------------------------
    def gpu_supported(self, op, age_ms) -> bool:
        """True when the draw tree is pure tile composition. Excluded: the
        leaderboard (not yet ported) and stage 2, whose `_draw_bar_rows` uses
        ImageDraw on `base` directly and so cannot be recorded."""
        if age_ms is None:
            return False
        op = _clamp01(op)
        fade = _clamp01(age_ms / FADE_MS) * op
        if fade <= 0.003:
            return False
        stage2 = getattr(self, "_stage2", False)
        if stage2 and age_ms > STAGE1_MS:
            return False
        stage1_a = fade
        if self.board is not None and stage1_a > 0.003:
            return False
        return True

    def capture_ops(self, w: int, h: int, op: float, age_ms: float):
        """Draw list for one outro frame, without needing the frozen frame.
        Only valid for op >= 1.0, where the base is pure black -- the fade-in
        frames composite over the frozen final gameplay frame, which lives on
        the composite thread, so those stay on the CPU path."""
        class _Shape:
            shape = (h, w, 4)
        return self._capture_ops(_Shape(), op, age_ms)

    def _capture_ops(self, rgb, op, age_ms):
        """Run the real layout against a recorder. Returns
        (frozen_alpha, [(img, x0, y0, alpha), ...]) where frozen_alpha is the
        alpha to draw the frozen gameplay frame at (0 => pure black base).

        The black wash over an opaque frame is `f*(255-A)//255`, i.e. the frame
        at alpha (255-A)/255 over black -- so on the GPU it is just the frozen
        frame drawn as one sprite, no multiply pass needed.
        """
        _dump = _os.environ.get("R3D_OUTRO_ELEMENTS")
        rec = DrawRecorder((rgb.shape[1], rgb.shape[0]), label=bool(_dump))
        fade = _clamp01(age_ms / FADE_MS) * op
        stage2 = getattr(self, "_stage2", False)
        panel_cx = self.center_cx_v * self.k if stage2 else self.cx_px
        self._draw_panel(rec, age_ms, fade, panel_cx)
        A = int(op * 255)
        frozen_a = 0.0 if op >= 1.0 else (255 - A) / 255.0
        if _dump and not getattr(CatchLazerResults, "_dumped", False):
            CatchLazerResults._dumped = True
            import sys as _ds
            print(f"\nOUTRO FRAME ELEMENTS  (age_ms={age_ms:.0f}, op={op:.3f}, "
                  f"stage2={getattr(self, '_stage2', False)})", file=_ds.stderr)
            print(f"  base: {'frozen frame @ alpha %.4f' % frozen_a if frozen_a else 'opaque black'}"
                  f"   canvas {rec.size[0]}x{rec.size[1]}", file=_ds.stderr)
            print(f"  {'#':>3} {'element (call site)':34s} {'size':>11} "
                  f"{'at':>13} {'alpha':>6} {'kpx':>7}", file=_ds.stderr)
            tot = 0
            for n, (img, x0, y0, a) in enumerate(rec.ops):
                nm = rec.names[n] if n < len(rec.names) else "?"
                kpx = img.width * img.height / 1000.0
                tot += kpx
                print(f"  {n:3d} {nm:34s} {img.width:5d}x{img.height:<5d} "
                      f"{x0:6d},{y0:<6d} {a:6.3f} {kpx:7.1f}", file=_ds.stderr)
            print(f"  {'':3} {'TOTAL':34s} {len(rec.ops):5d} elements"
                  f"{'':20} {tot:7.1f} kpx composited", file=_ds.stderr)
        return frozen_a, rec.ops

    def _verify_capture(self, rgb, op, age_ms, cpu_out):
        """Replay the captured draw list through PIL and compare with the real
        CPU frame. Proves the recorder captured EVERY composite, before any GPU
        code is trusted -- a missing element would show up as a big diff here
        rather than as a subtle bug on screen later."""
        import sys as _vs, numpy as _vnp
        if not self.gpu_supported(op, age_ms):
            print(f"CAPTURE-VERIFY  frame not GPU-supported "
                  f"(age={age_ms:.0f} op={op:.3f} stage2="
                  f"{getattr(self, '_stage2', False)} board="
                  f"{self.board is not None})", file=_vs.stderr)
            return
        frozen_a, ops = self._capture_ops(rgb, op, age_ms)
        size = (rgb.shape[1], rgb.shape[0])
        if frozen_a <= 0.0:
            rep = Image.new("RGBA", size, (0, 0, 0, 255))
        else:
            A = int(op * 255)
            lut = [(v * (255 - A) + 127) // 255 for v in range(256)]
            rep = Image.fromarray(rgb, "RGBA").point(lut * 3 + [255] * 256)
        for img, x0, y0 in ((o[0], o[1], o[2]) for o in ops):
            pass
        for img, x0, y0, a in ops:
            im = img
            if a < 0.997:
                ach = img.getchannel("A").point(lambda v: int(v * a))
                im = img.copy()
                im.putalpha(ach)
            rep.alpha_composite(im, (x0, y0))
        a1 = _vnp.asarray(rep)[..., :3].astype(int)
        a2 = (cpu_out[..., :3].astype(int) if cpu_out.ndim == 3
              and cpu_out.shape[2] == 4 else cpu_out.astype(int))
        d = _vnp.abs(a1 - a2)
        print(f"CAPTURE-VERIFY  {len(ops)} recorded ops, frozen_alpha="
              f"{frozen_a:.4f}  max|diff|={int(d.max())}  "
              f"px>0={int((d.max(axis=2) > 0).sum())}  "
              f"=> {'IDENTICAL' if d.max() == 0 else 'MISMATCH'}",
              file=_vs.stderr)

    def _pose_signature(self, op, age_ms):
        """Tuple of EVERY animated parameter render_frame consumes for
        (op, age_ms) — all of them clamped easings, so the tuple saturates
        when the animation does. None disables the memo (fail-safe)."""
        try:
            op = _clamp01(op)
            fade = _clamp01(age_ms / FADE_MS) * op
            stage2 = getattr(self, "_stage2", False)
            open_p = ease_out_quint((age_ms - STAGE1_MS) / OPEN_MS) \
                if (stage2 and age_ms > STAGE1_MS) else 0.0
            stage1_a = fade * (1.0 - _clamp01((age_ms - STAGE1_MS)
                                              / (OPEN_MS * 0.6))) \
                if stage2 else fade
            lb = None
            if self.board is not None and fade > 0.003 and stage1_a > 0.003:
                lb = tuple(self._lb_slide(c.stagger, age_ms)
                           for c in (getattr(self.board, "cards", []) or []))
            # featured panel internals (arc sweep / grade punch / score roll)
            sweep = ease_out_cubic((age_ms - SWEEP_DELAY_MS) / SWEEP_MS) \
                if age_ms > SWEEP_DELAY_MS else 0.0
            arc_bucket = round(self.target_arc * sweep, 3)
            score_val = int(round(int(self.meta.score) * sweep))
            badge_start = SWEEP_DELAY_MS + SWEEP_MS - 130.0
            grade_p = ease_out_cubic((age_ms - badge_start) / BADGE_MS) \
                if age_ms >= badge_start else None
            unfold = tuple(self._panel_unfold(age_ms, i) for i in range(3)) \
                if (stage2 and age_ms > STAGE1_MS) else None
            return (op, fade, open_p, stage1_a, lb, arc_bucket, score_val,
                    grade_p, unfold)
        except Exception:  # noqa: BLE001 — memo is best-effort, never breaks
            return None

    def _blit(self, base, img, cx, top_y, a) -> float:
        """Paste centred at cx with the image TOP at top_y; return height."""
        _paste(base, img, cx, top_y + img.height / 2.0, a)
        return float(img.height)

    def _draw_panel(self, base, age_ms, a, cx=None, skip_bg=False) -> None:
        import os as _os, time as _t
        _pm = self._RP if _os.environ.get("R3D_RESULTS_PERF") else None
        _mt = [_t.perf_counter()] if _pm is not None else None

        def _mk(key):
            if _pm is not None:
                now = _t.perf_counter()
                _pm[key] += now - _mt[0]
                _mt[0] = now

        if _pm is not None:
            _pm["p_n"] += 1
        k = self.k
        if cx is None:
            cx = self.cx_px
        if not skip_bg:                  # already in the reused base
            _paste(base, self.panel_img, cx, self.panel_cy_px, a)
        _mk("p_bg")
        top = (self.panel_cy_v - PANEL_H / 2.0 + 40.0) * k
        y = top
        # header: avatar + name (centred group)
        av = float(self.avatar_img.width)
        nw = float(self.name_img.width)
        group_w = av + 12 * k + nw
        gx = cx - group_w / 2.0
        _paste(base, self.avatar_img, gx + av / 2.0, y + av / 2.0, a)
        _paste(base, self.name_img, gx + av + 12 * k + nw / 2.0,
               y + av / 2.0, a)
        _mk("p_header")
        y += av + 16 * k
        y += self._blit(base, self.title_img, cx, y, a) + 4 * k
        y += self._blit(base, self.artist_img, cx, y, a) + 20 * k
        _mk("p_titles")
        # accuracy circle: base ring + sweeping arc + punching grade letter
        acc_d = ACC_DISP * k
        circ_cy = y + acc_d / 2.0
        _paste(base, self.acc_base_img, cx, circ_cy, a)
        _mk("p_accbase")
        self._draw_acc_arc(base, age_ms, cx, circ_cy, a)
        _mk("p_arc")
        self._draw_grade(base, age_ms, cx, circ_cy, a)
        _mk("p_grade")
        y += acc_d + 6 * k
        # score (rolls with the sweep)
        self._roll_score(age_ms)
        y += self._blit(base, self._score_img, cx, y, a) + 12 * k
        _mk("p_score")
        # played-mod badge row (nomod → no row, layout unchanged)
        mod_h = self._draw_mod_row(base, cx, y, a)
        _mk("p_mods")
        if mod_h > 0.0:
            y += mod_h + 14 * k
        # star / diff / creator centred row
        y += self._draw_star_row(base, cx, y, a) + 22 * k
        _mk("p_star")
        # stats grid (catch: ACC/COMBO/PP then FRUIT/DROP/DROPLET/MISS)
        y += self._draw_grid_row(base, self.grid_a, cx, y, a,
                                 PANEL_W * 0.86) + 16 * k
        self._draw_grid_row(base, self.grid_b, cx, y, a, PANEL_W * 0.86)
        _mk("p_grids")
        # date row anchored to the panel bottom (std footer position)
        _paste(base, self.date_img, cx,
               (self.panel_cy_v + PANEL_H / 2.0 - 40.0) * k, a)
        _mk("p_date")

    def _draw_acc_arc(self, base, age_ms, cx, cy, a) -> None:
        sweep = ease_out_cubic((age_ms - SWEEP_DELAY_MS) / SWEEP_MS) \
            if age_ms > SWEEP_DELAY_MS else 0.0
        prog = self.target_arc * sweep
        bucket = round(prog, 3)
        import os as _os2, time as _t2
        _pb = self._RP if _os2.environ.get("R3D_RESULTS_PERF") else None
        _tb = _t2.perf_counter() if _pb is not None else 0.0
        _shared = _ARC_SHARED.get((max(int(ACC_DISP * self.k), 64), bucket))
        if _shared is not None:
            if bucket != self._arc_bucket:
                self._arc_img, self._arc_bucket = _shared, bucket
            if _pb is not None:
                _pb["p_archit"] = _pb.get("p_archit", 0) + 1
            _paste(base, self._arc_img, cx, cy, a)
            return
        if bucket != self._arc_bucket or self._arc_img is None:
            # Bake from the BUCKET, not the raw prog. Keyed on round(prog,3)
            # but baked from prog, the cached arc depended on which prog value
            # arrived FIRST within a bucket -- i.e. on evaluation order. That is
            # deterministic single-threaded only by accident of fixed ordering,
            # and it is what made the parallel outro diverge. Baking from the key
            # makes the cache a pure function of the key, so any ordering (and
            # any worker count) gives identical pixels.
            self._arc_img = bake_accuracy_arc(int(ACC_DISP * self.k), bucket)
            self._arc_bucket = bucket
            if _pb is not None:
                # the BAKE, separated from the paste: it only runs when the
                # sweep crosses a 0.001 bucket, so its per-frame share depends
                # on how fast the arc is moving
                _pb["p_arcbake"] += _t2.perf_counter() - _tb
        _paste(base, self._arc_img, cx, cy, a)

    def _draw_grade(self, base, age_ms, cx, cy, a) -> None:
        badge_start = SWEEP_DELAY_MS + SWEEP_MS - 130.0
        if age_ms < badge_start:
            return
        p = ease_out_cubic((age_ms - badge_start) / BADGE_MS)
        scale = _lerp(1.42, 1.0, p)
        ga = a * _clamp01(p * 1.4)
        img = self.grade_img
        if abs(scale - 1.0) > 0.005:
            img = img.resize((max(int(img.width * scale), 1),
                              max(int(img.height * scale), 1)), Image.LANCZOS)
        _paste(base, img, cx, cy, ga)

    def _roll_score(self, age_ms) -> None:
        sweep = ease_out_cubic((age_ms - SWEEP_DELAY_MS) / SWEEP_MS) \
            if age_ms > SWEEP_DELAY_MS else 0.0
        val = int(round(int(self.meta.score) * sweep))
        if val != self._score_val:
            self._score_text(val)

    def _draw_mod_row(self, base, cx, top_y, a) -> float:
        if not self.mod_pills:
            return 0.0
        gap = MOD_PILL_GAP_V * self.k
        total = sum(p.width for p in self.mod_pills) \
            + gap * (len(self.mod_pills) - 1)
        h = max(p.height for p in self.mod_pills)
        x = cx - total / 2.0
        for p in self.mod_pills:
            _paste(base, p, x + p.width / 2.0, top_y + h / 2.0, a)
            x += p.width + gap
        return float(h)

    def _draw_star_row(self, base, cx, top_y, a) -> float:
        gap = 14 * self.k
        rows = [self.star_pill, self.diff_img, self.creator_img]
        total = sum(r.width for r in rows) + gap * (len(rows) - 1)
        h = max(r.height for r in rows)
        x = cx - total / 2.0
        for r in rows:
            _paste(base, r, x + r.width / 2.0, top_y + h / 2.0, a)
            x += r.width + gap
        return float(h)

    def _draw_grid_row(self, base, cells, cx, top_y, a, span_virtual) -> float:
        k = self.k
        span = span_virtual * k
        n = len(cells)
        col_w = span / n
        x0 = cx - span / 2.0
        row_h = 0.0
        for i, (label, value) in enumerate(cells):
            ccx = x0 + col_w * (i + 0.5)
            _paste(base, label, ccx, top_y + label.height / 2.0, a)
            _paste(base, value, ccx,
                   top_y + label.height + 6 * k + value.height / 2.0, a)
            row_h = max(row_h, label.height + 6 * k + value.height)
        return row_h

    # -- flank leaderboard (the kept lb_cards board, std layout/timing) -----------

    def _lb_slide(self, i: int, age_ms: float) -> tuple[float, float]:
        """std's _lb_slide: card i eases in over LB_SLIDE_MS (OutQuint) from
        LB_SLIDE_OFFSET virtual px outward, starting at LB_SLIDE_START_MS +
        i*LB_SLIDE_STAGGER_MS. Returns (outward_offset_virtual, alpha)."""
        t = age_ms - LB_SLIDE_START_MS - i * LB_SLIDE_STAGGER_MS
        p = ease_out_quint(t / LB_SLIDE_MS)
        return (1.0 - p) * LB_SLIDE_OFFSET, p

    def _draw_board(self, base, age_ms, a, dx: float = 0.0) -> None:
        """`dx` (px) shifts the whole board with the featured panel during
        the stage-2 slide — std positions its flanks off the moving panel
        edges, so the cards + banner track the panel while fading out."""
        board = self.board
        for c in getattr(board, "cards", []) or []:
            off, ca = self._lb_slide(c.stagger, age_ms)
            if ca <= 0.003:
                continue
            x = c.cx + c.out_dir * off * self.k + dx
            _paste(base, c.img, x, c.cy, a * ca)
        banner = getattr(board, "banner", None)
        if banner is not None:
            bimg, _bx, _by = banner
            _paste(base, bimg, self.cx_px + dx,
                   self.panel_top_px - bimg.height / 2.0 - 6 * self.k, a)

    # -- stats panels (stage 2) ----------------------------------------------------

    def _panel_unfold(self, age_ms: float, idx: int) -> float:
        """std's _panel_unfold: panel `idx` starts at STAGE1_MS + 120 +
        idx·STAGGER_MS and eases open over OPEN_MS (OutQuint)."""
        start = STAGE1_MS + 120.0 + idx * STAGGER_MS
        return ease_out_quint((age_ms - start) / OPEN_MS) \
            if age_ms > start else 0.0

    def _draw_stats_panels(self, base, age_ms, op) -> None:
        """The three stats panels unfolding from the right, staggered (std's
        _draw_stats_panels): each panel bg is width-squashed by its unfold
        progress, anchored at STATS_X0 (growing rightward), the content
        fading in at full-width position once the panel is ~55% open."""
        k = self.k
        top0 = self.panel_cy_v - PANEL_H / 2.0
        drawers = (self._draw_perf, self._draw_combo, self._draw_catches)
        for idx, drawer in enumerate(drawers):
            s = self._panel_unfold(age_ms, idx)
            if s <= 0.003:
                continue
            pcy = (top0 + self.STATS_H * (idx + 0.5) + idx * 24.0) * k
            left = self.STATS_X0 * k
            pw, ph = self.stats_panel_img.size
            drawn_w = max(int(pw * s), 1)
            img = self.stats_panel_img if drawn_w >= pw else \
                self.stats_panel_img.resize((drawn_w, ph), Image.BILINEAR)
            _paste(base, img, left + drawn_w / 2.0, pcy,
                   min(s * 1.3, 1.0) * op)
            content_a = _clamp01((s - 0.55) / 0.45) * op
            if content_a > 0.01:
                full_cx = (self.STATS_X0 + self.STATS_W / 2.0) * k
                drawer(base, full_cx,
                       (top0 + idx * (self.STATS_H + 24.0)) * k, content_a)

    def _draw_bar_rows(self, base, d, cx, top_y, a, title_img, rows,
                       foot_img, bar_x_off: float, right_reserve: float
                       ) -> None:
        """Shared std perf-panel layout: title top-left, label + track +
        fill + right-aligned value per row, footer centred at the bottom.
        `rows` = (label_img, value_img, frac, colour)."""
        k = self.k
        pad = 26 * k
        left = cx - self.STATS_W * k / 2.0 + pad
        right = cx + self.STATS_W * k / 2.0 - pad
        _paste(base, title_img, left + title_img.width / 2.0,
               top_y + 22 * k + title_img.height / 2.0, a)
        th = title_img.height
        y = top_y + 22 * k + th + 20 * k
        bar_x = left + bar_x_off * k
        bar_w = right - bar_x - right_reserve * k
        bar_h = 16 * k
        row_gap = (self.STATS_H * k - 22 * k - th - 20 * k - 40 * k) / \
            max(len(rows), 1)
        for i, (label, val, frac, col) in enumerate(rows):
            ry = y + i * row_gap
            cyy = ry + bar_h / 2.0
            _paste(base, label, left + label.width / 2.0, cyy, a)
            rc = tuple(int(round(c * 255)) for c in col)
            d.rectangle([bar_x, ry, bar_x + bar_w, ry + bar_h],
                        fill=(51, 56, 71, int(0.7 * a * 255)))
            fw = max(bar_w * _clamp01(frac), 2.0)
            d.rectangle([bar_x, ry, bar_x + fw, ry + bar_h],
                        fill=(*rc, int(0.95 * a * 255)))
            _paste(base, val, right - val.width / 2.0, cyy, a)
        _paste(base, foot_img, cx,
               top_y + self.STATS_H * k - 20 * k - foot_img.height / 2.0, a)

    def _draw_perf(self, base, cx, top_y, a) -> None:
        """PERFORMANCE: accuracy / combo-vs-map-max / achieved-vs-SS-pp bars
        (std's perf-bar layout, catch's real numbers)."""
        d = ImageDraw.Draw(base, "RGBA")
        self._draw_bar_rows(base, d, cx, top_y, a, self.perf_title,
                            self._perf_rows, self.perf_foot,
                            bar_x_off=130.0, right_reserve=60.0)

    def _draw_combo(self, base, cx, top_y, a) -> None:
        """COMBO progression (the sim's checkpoint series) / DIFFICULTY
        strain fallback — the pre-baked area chart in std's timing-panel
        slot (title top-left, chart body, stat footer)."""
        k = self.k
        pad = 26 * k
        left = cx - self.STATS_W * k / 2.0 + pad
        _paste(base, self.combo_title, left + self.combo_title.width / 2.0,
               top_y + 22 * k + self.combo_title.height / 2.0, a)
        th = self.combo_title.height
        chart = self.combo_chart
        if chart is not None:
            _paste(base, chart, cx,
                   top_y + 22 * k + th + 16 * k + chart.height / 2.0, a)
            foot_cy = top_y + 22 * k + th + 16 * k + chart.height + 22 * k
        else:
            foot_cy = top_y + self.STATS_H * k / 2.0
        _paste(base, self.combo_foot, cx, foot_cy, a)

    def _draw_catches(self, base, cx, top_y, a) -> None:
        """CATCHES: per-type caught/total bars in the flank-card palette +
        the hyperdash-usage footer."""
        d = ImageDraw.Draw(base, "RGBA")
        self._draw_bar_rows(base, d, cx, top_y, a, self.catch_title,
                            self._catch_rows, self.catch_foot,
                            bar_x_off=130.0, right_reserve=110.0)
