import os
import sys


def _run_metal() -> None:
    """EXPERIMENTAL macOS Metal backend: R3D_METAL=1 on darwin only.

    `osu_catch_renderer/_metal` is a self-contained copy of the renderer (the
    2026-09 Metal fork, frozen on 4bf5909) that draws through a Swift/Metal
    dylib instead of GL. It is a SEPARATE package on purpose: nothing in the GL
    renderer imports it, so with the flag unset this file's only effect is the
    two imports above and output is byte-identical to before.

    Not a drop-in replacement: it predates the feat/catch-ship fixes and its
    GPU HUD is an approximation of the CPU one, so Metal output is close to GL
    but NOT identical (see _metal/README.md). Any failure to load or render
    falls back to the normal GL path in a fresh process.
    """
    try:
        from osu_catch_renderer._metal.cli import main as metal_main
        for k, v in (("R3D_METAL_HUD", "1"), ("R3D_METAL_YUV", "1"),
                     ("R3D_MAC_SOCKET_PIPE", "1")):
            os.environ.setdefault(k, v)
        print("[catch] backend: METAL (experimental, R3D_METAL=1)",
              file=sys.stderr, flush=True)
        rc = metal_main()
    except SystemExit as e:
        rc = e.code
    except BaseException as e:  # noqa: BLE001 - never let Metal lose a render
        print(f"[catch] Metal backend raised {e!r}", file=sys.stderr, flush=True)
        rc = 1
    if not rc:
        raise SystemExit(0)
    print(f"[catch] Metal backend failed (rc={rc}) -> re-running on GL",
          file=sys.stderr, flush=True)
    env = dict(os.environ, R3D_METAL="0")
    os.execve(sys.executable,
              [sys.executable, "-m", "osu_catch_renderer"] + sys.argv[1:], env)


if sys.platform == "darwin" and os.environ.get("R3D_METAL") == "1":
    _run_metal()

from osu_catch_renderer.cli import main  # noqa: E402

raise SystemExit(main())
