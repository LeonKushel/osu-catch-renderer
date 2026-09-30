"""Shared 8-bit store helper for catch's PER-FRAME composites.

`R3D_CATCH_ROUND=1` rounds to nearest instead of truncating.

WHY: `.astype(np.uint8)` TRUNCATES. On a blend that is a BIASED estimator — it
loses on average half a level on every blended pixel — so every element composited
this way renders systematically DARK. taiko had the same defect in two modules and
fixing it was measurable: it took a correct GPU flashlight port from "+1 on
3,850,533 channel values" to "0.0001%% of pixels", and the same for its break
overlay. catch's own `flashlight.py:188` already uses `np.rint`, which is exactly
why catch's GPU flashlight port measured 49 differing values where taiko's
measured 3.85 million.

Default OFF because it CHANGES shipped output (slightly brighter, and correct).
Red's call, per `misc/2026-09-29-FOR-RED-CLAUDE-flashlight-GPU-quad-two-options.md`.

NOT covered here: the three ALPHA-channel truncations (`argon_hud.py:427`,
`argon_hud.py:476`, `argon_counter.py:356`). Rounding those changes compositing
semantics rather than brightness, so they are a separate decision.
"""
import os

import numpy as np

ROUND = bool(os.environ.get("R3D_CATCH_ROUND"))


def to8_clipped(x):
    """uint8 store for a value ALREADY clipped to 0..255.

    Deliberately does not re-clip: every call site already does, and adding a
    second full-array pass would cost performance in a change meant to be free.
    """
    return np.rint(x).astype(np.uint8) if ROUND else x.astype(np.uint8)
