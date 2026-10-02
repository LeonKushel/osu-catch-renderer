# `_metal` — experimental macOS Metal backend for catch

Enabled only when **`R3D_METAL=1` on darwin** (see `osu_catch_renderer/__main__.py`).
With the flag unset nothing here is imported and catch output is byte-identical to
before.

## What it is

A self-contained copy of the catch renderer — the 2026-09 Metal fork, frozen on
`4bf5909` — that draws through a Swift/Metal dylib (`render/metal/`) instead of GL.
It is a separate package rather than a merge into `render/render.py` on purpose:
the fork changed ~2,600 lines across files `feat/catch-ship` has since changed too,
and a hand merge would have put the GL path at risk. This way the GL path is
untouched.

## What it is NOT

* **Not identical to GL.** The GPU HUD (health bar, wedges, progress strip) is an
  approximation of the CPU one: ~1.5% of pixels differ, confined to those elements.
* **Not current.** It predates the `feat/catch-ship` fixes. Seen so far: a failed
  play renders 4073 frames here vs 4069 on GL.
* **Not for Linux/Windows.** `__main__` only takes this path on darwin.

## Measured (M1 Max, bundle 0.1.22's Python 3.12, via `r3d_dispatch.py`, 1280x720/60)

| map | GL | Metal |
|---|---|---|
| 30 s plain | 12.8 s | 6.4 s |
| 60 s flashlight | 30.2 s | 12.5 s |
| 87 s failed play | 25.7 s | 11.0 s |
| 134 s DT | 17.3 s | 9.6 s |

Deterministic run to run (identical mp4). Right way up (frames match GL as-read, not
mirrored). If the dylib cannot load or the render fails, `__main__` re-runs the job
on GL in a fresh process — verified by making the dylib unloadable: output was
byte-identical to a normal GL render.

## Flags

`R3D_METAL=1` turns it on; `__main__` then defaults `R3D_METAL_HUD=1`,
`R3D_METAL_YUV=1`, `R3D_MAC_SOCKET_PIPE=1`. `R3D_METAL_FL_UNDER_HUD` defaults to on
(flashlight is drawn UNDER the in-pass HUD, as GL does; `=0` restores the fork's
original order, which blacks out the health bar on flashlight maps).

## The dylib

`render/metal/libr3dmetal.dylib` is arm64, built from `render/metal/src/*.swift` with
`render/metal/build.sh` (Apple Swift 6.4). A rebuild from those sources gives a
different file hash but **byte-identical videos** on the plain, flashlight and
failed-play maps above — rebuild it yourself rather than trusting the binary.
