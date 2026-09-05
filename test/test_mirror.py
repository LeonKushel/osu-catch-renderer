"""Regression tests for lazer Catch Mirror auto-detection (scene.CatchSim._maybe_apply_mirror).

Mirror is a lazer-only mod that flips the playfield horizontally. lazer's legacy .osr
export writes mods:0 in the stable bitfield (the real mod set lives in an appended
LegacyReplaySoloScoreInfo block osrparse does not decode), so without detection every
fruit is placed on the wrong side and the honesty guard false-rejects a legit replay.

_maybe_apply_mirror uses no instance state, so we drive it directly with a dummy self
(None) and lightweight beatmap/meta stubs.
"""
from types import SimpleNamespace

from osu_catch_renderer.beatmap.models import CatchObject, CatchFrame, ObjType
from osu_catch_renderer.render.scene import CatchSim

LAZER = 30000019          # a lazer legacy-export version (>= 30000000)
STABLE = 20210520         # a stable client version


class _BM:
    """Minimal stand-in for CatchBeatmap: the detector only reads .cs and .objects."""
    def __init__(self, cs, objects):
        self.cs = cs
        self.objects = objects


def _objects(n=120):
    """n fruit spread across the playfield, one flagged hyperdash with a target."""
    objs = []
    for i in range(n):
        x = float((i * 37) % 500 + 6)          # 6..505, well spread
        hyper = (i == 10)
        objs.append(CatchObject(
            time_ms=i * 100, x=x, kind=ObjType.FRUIT,
            hyperdash=hyper,
            hyper_target_x=(400.0 if hyper else None),
        ))
    return objs


def _frames(objs, mirror):
    """One frame per object placing the catcher exactly on the (optionally mirrored)
    fruit, so identity vs 512-x alignment is unambiguous. Extra tail frame so every
    object is inside the frame span."""
    fr = [CatchFrame(time_ms=o.time_ms, x=(512.0 - o.x) if mirror else o.x, dashing=False)
          for o in objs]
    fr.append(CatchFrame(time_ms=objs[-1].time_ms + 100, x=fr[-1].x, dashing=False))
    return fr


def test_lazer_mirror_detected_and_flipped():
    objs = _objects()
    orig = [(o.x, o.hyper_target_x) for o in objs]
    bm = _BM(cs=4.0, objects=list(objs))
    frames = _frames(objs, mirror=True)          # catcher tracks the mirrored layout
    CatchSim._maybe_apply_mirror(None, bm, frames, SimpleNamespace(game_version=LAZER))
    # every object flipped x -> 512 - x
    for o, (ox, _) in zip(bm.objects, orig):
        assert o.x == 512.0 - ox


def test_mirrored_hyper_target_is_flipped():
    objs = _objects()
    bm = _BM(cs=4.0, objects=list(objs))
    frames = _frames(objs, mirror=True)
    CatchSim._maybe_apply_mirror(None, bm, frames, SimpleNamespace(game_version=LAZER))
    hyper = [o for o in bm.objects if o.hyperdash][0]
    # hyper_target_x is an ABSOLUTE coord and must mirror too (Aussie review fix)
    assert hyper.hyper_target_x == 512.0 - 400.0


def test_lazer_nomod_unchanged():
    objs = _objects()
    orig = [o.x for o in objs]
    bm = _BM(cs=4.0, objects=list(objs))
    frames = _frames(objs, mirror=False)         # catcher tracks the real layout
    CatchSim._maybe_apply_mirror(None, bm, frames, SimpleNamespace(game_version=LAZER))
    assert [o.x for o in bm.objects] == orig


def test_stable_replay_never_flipped():
    """Even geometry that would look 'mirrored' must not be touched for a stable
    replay: Catch Mirror cannot legitimately exist there."""
    objs = _objects()
    orig = [o.x for o in objs]
    bm = _BM(cs=4.0, objects=list(objs))
    frames = _frames(objs, mirror=True)          # would trip the heuristic if allowed
    CatchSim._maybe_apply_mirror(None, bm, frames, SimpleNamespace(game_version=STABLE))
    assert [o.x for o in bm.objects] == orig
