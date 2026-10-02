"""ctypes binding for the R3D Metal backend. The frame comes back as a numpy
view over GPU-rendered shared memory -- no copy."""
import ctypes, os
import numpy as np

_LIB = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "libr3dmetal.dylib"))
_LIB.r3dm_create.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
_LIB.r3dm_create.restype = ctypes.c_int32
_LIB.r3dm_info.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32),
                           ctypes.POINTER(ctypes.c_int32)]
_LIB.r3dm_info.restype = ctypes.c_int32
_LIB.r3dm_device_name.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32]
_LIB.r3dm_device_name.restype = ctypes.c_int32
_LIB.r3dm_begin.argtypes = [ctypes.c_int32] + [ctypes.c_float]*4
_LIB.r3dm_begin.restype = ctypes.c_int32
_LIB.r3dm_commit.argtypes = [ctypes.c_int32]
_LIB.r3dm_commit.restype = ctypes.c_int32
_LIB.r3dm_acquire.argtypes = [ctypes.c_int32, ctypes.c_int32]
_LIB.r3dm_acquire.restype = ctypes.c_void_p
_LIB.r3dm_yuv_init.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
_LIB.r3dm_yuv_init.restype = ctypes.c_int32
_LIB.r3dm_yuv_len.argtypes = [ctypes.c_int32]
_LIB.r3dm_yuv_len.restype = ctypes.c_int32
_LIB.r3dm_yuv_input.argtypes = [ctypes.c_int32]
_LIB.r3dm_yuv_input.restype = ctypes.c_void_p
_LIB.r3dm_yuv_submit.argtypes = [ctypes.c_int32]
_LIB.r3dm_yuv_submit.restype = ctypes.c_int32
_LIB.r3dm_yuv_acquire.argtypes = [ctypes.c_int32, ctypes.c_int32]
_LIB.r3dm_yuv_acquire.restype = ctypes.c_void_p
_LIB.r3dm_yuv_from_ring.argtypes = [ctypes.c_int32]
_LIB.r3dm_yuv_from_ring.restype = ctypes.c_void_p
_LIB.r3dm_commit_yuv.argtypes = [ctypes.c_int32]
_LIB.r3dm_commit_yuv.restype = ctypes.c_int32
_LIB.r3dm_yuv_ring_acquire.argtypes = [ctypes.c_int32, ctypes.c_int32,
                                       ctypes.c_int32]
_LIB.r3dm_yuv_ring_acquire.restype = ctypes.c_void_p
_LIB.r3dm_flash_inline.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)]
_LIB.r3dm_flash_inline.restype = ctypes.c_int32
_LIB.r3dm_tex_create.argtypes = [ctypes.c_int32, ctypes.c_int32,
                                 ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32]
_LIB.r3dm_tex_create.restype = ctypes.c_int32
_LIB.r3dm_draw.argtypes = [ctypes.c_int32, ctypes.c_void_p, ctypes.c_int32,
                           ctypes.POINTER(ctypes.c_int32), ctypes.c_int32,
                           ctypes.c_int32]
_LIB.r3dm_draw.restype = ctypes.c_int32
_LIB.r3dm_death_init.argtypes = [ctypes.c_int32]
_LIB.r3dm_death_init.restype = ctypes.c_int32
_LIB.r3dm_death_src.argtypes = [ctypes.c_int32]
_LIB.r3dm_death_src.restype = ctypes.c_void_p
_LIB.r3dm_death_run.argtypes = [ctypes.c_int32] + [ctypes.c_float]*8
_LIB.r3dm_death_run.restype = ctypes.c_void_p
_LIB.r3dm_hp_init.argtypes = [ctypes.c_int32]
_LIB.r3dm_hp_init.restype = ctypes.c_int32
_LIB.r3dm_hp_sdf.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                             ctypes.c_int32, ctypes.c_void_p]
_LIB.r3dm_hp_sdf.restype = ctypes.c_int32
_LIB.r3dm_hp_src.argtypes = [ctypes.c_int32]
_LIB.r3dm_hp_src.restype = ctypes.c_void_p
_LIB.r3dm_hp_run.argtypes = [ctypes.c_int32, ctypes.c_void_p]
_LIB.r3dm_hp_run.restype = ctypes.c_void_p
_LIB.r3dm_pill_inline.argtypes = [ctypes.c_int32, ctypes.c_void_p]
_LIB.r3dm_pill_inline.restype = ctypes.c_int32
_LIB.r3dm_hp_inline.argtypes = [ctypes.c_int32, ctypes.c_void_p]
_LIB.r3dm_hp_inline.restype = ctypes.c_int32
_LIB.r3dm_destroy.argtypes = [ctypes.c_int32]
_LIB.r3dm_destroy.restype = ctypes.c_int32


class MetalRenderer:
    def __init__(self, width, height, ring=3, copy_out=True):
        self.width, self.height = width, height
        self.copy_out = copy_out
        self.id = _LIB.r3dm_create(width, height, ring)
        if self.id < 0:
            raise RuntimeError("r3dm_create failed (no Metal device?)")
        bpr = ctypes.c_int32(); blen = ctypes.c_int32()
        _LIB.r3dm_info(self.id, ctypes.byref(bpr), ctypes.byref(blen))
        self.bpr, self.buf_len = bpr.value, blen.value
        buf = ctypes.create_string_buffer(256)
        _LIB.r3dm_device_name(self.id, buf, 256)
        self.device = buf.value.decode()
        # no row padding on M1 Max at 1920 (align 16, 1920*4 = 7680)
        self.padded = self.bpr != self.width * 4

    def begin(self, rgba=(0.0, 0.0, 0.0, 1.0)):
        if _LIB.r3dm_begin(self.id, *[float(x) for x in rgba]) != 0:
            raise RuntimeError("r3dm_begin failed")

    def commit(self):
        r = _LIB.r3dm_commit(self.id)
        if r < 0:
            raise RuntimeError("r3dm_commit failed")
        return bool(r)

    def acquire(self, force=False):
        """Zero-copy numpy view of the OLDEST completed frame, or None while the
        ring fills (force=True drains the tail). Valid until that ring slot is
        reused, i.e. RING frames later -- same contract as gl._HOST_POOL."""
        p = _LIB.r3dm_acquire(self.id, 1 if force else 0)
        if not p:
            return None
        arr = np.ctypeslib.as_array(
            (ctypes.c_uint8 * self.buf_len).from_address(p))
        if self.padded:
            view = arr.reshape((self.height, self.bpr))[:, :self.width*4] \
                      .reshape((self.height, self.width, 4))
        else:
            view = arr.reshape((self.height, self.width, 4))
        if self.copy_out:
            # LIFETIME: the ring slot handed out here is reused by the NEXT
            # begin() -- when acquire fires, head-tail == ring, so
            # head % ring == (tail-1) % ring. The zero-copy view is therefore
            # only valid until the next frame starts, which is useless for a
            # pipeline that queues frames through a composite thread and a
            # 12-deep writer queue. Copying costs ~0.19 ms (still ~10x better
            # than the GL readback's 1.87 ms). The proper fix is an explicit
            # release so a slot is not reused while a consumer holds it.
            return view.copy()
        return view

    def tex_create(self, rgba, mipmaps=True):
        """Upload an HxWx4 uint8 array; returns a texture id."""
        a = np.ascontiguousarray(rgba, dtype=np.uint8)
        h, w = a.shape[0], a.shape[1]
        t = _LIB.r3dm_tex_create(self.id, w, h,
                                 a.ctypes.data_as(ctypes.c_void_p),
                                 1 if mipmaps else 0)
        if t < 0:
            raise RuntimeError("r3dm_tex_create failed")
        return t

    # 14 floats/instance, same layout as gl.py's _I_STRIDE
    STRIDE = 14

    def draw(self, inst, tex_ids, additive=False):
        """inst: (N,14) float32. tex_ids: list mapping unit -> texture id."""
        a = np.ascontiguousarray(inst, dtype=np.float32)
        n = a.shape[0]
        ids = (ctypes.c_int32 * max(1, len(tex_ids)))(*tex_ids)
        r = _LIB.r3dm_draw(self.id, a.ctypes.data_as(ctypes.c_void_p), n,
                           ids, len(tex_ids), 1 if additive else 0)
        if r != 0:
            raise RuntimeError(f"r3dm_draw failed ({r})")

    # ---- GPU apply_death ---------------------------------------------------
    def death_init(self):
        if _LIB.r3dm_death_init(self.id) != 0:
            raise RuntimeError("r3dm_death_init failed")
        self._dsrc = _LIB.r3dm_death_src(self.id)
        if not self._dsrc:
            raise RuntimeError("r3dm_death_src returned NULL")

    def _wrap(self, ptr):
        arr = np.ctypeslib.as_array((ctypes.c_uint8 * self.buf_len).from_address(ptr))
        if self.padded:
            return arr.reshape((self.height, self.bpr))[:, :self.width*4] \
                      .reshape((self.height, self.width, 4))
        return arr.reshape((self.height, self.width, 4))

    def death_src_view(self):
        """Writable HxWx4 view of the GPU-visible input buffer -- write the
        composited frame here, then call death_run."""
        return self._wrap(self._dsrc)

    def death_run(self, mul, red_add, coeffs):
        p = _LIB.r3dm_death_run(self.id, float(mul), float(red_add),
                                *[float(x) for x in coeffs])
        if not p:
            raise RuntimeError("r3dm_death_run failed")
        # COPY, do not hand out the view. r3dm_death_run owns ONE destination
        # buffer and reuses it every death frame, but the frame it returns is
        # queued downstream (composite -> writer, up to 12 deep) and read much
        # later -- the next death frame would overwrite a frame still in
        # flight. Measured: this is what made death frames differ run-to-run
        # and between HUD configs (22-27 dB on a warped frame). Same bug the
        # sprite ring's copy-on-acquire fixes; the death path had no ring.
        # ~0.3 ms per death frame, and only death frames pay it.
        return np.array(self._wrap(p))

    # ---- GPU RGBA -> yuv420p (ffmpeg then runs no swscale) ------------------
    def yuv_init(self, ring=None, depth=None):
        """Ring of (input, output, command buffer) triples. Ringing is what lets
        submit() skip waitUntilCompleted: with one input buffer the next frame's
        write would race the in-flight kernel reading it."""
        ring = int(os.environ.get("R3D_YUV_RING", ring or 6) or 6)
        depth = int(os.environ.get("R3D_YUV_DEPTH", depth or 2) or 2)
        if _LIB.r3dm_yuv_init(self.id, ring, depth) != 0:
            raise RuntimeError("r3dm_yuv_init failed")
        self._ylen = int(_LIB.r3dm_yuv_len(self.id))
        if self._ylen <= 0:
            raise RuntimeError("r3dm_yuv_len failed")
        self._yring, self._ydepth = ring, depth

    def yuv_submit(self, frame) -> bool:
        """Copy one composited RGBA frame into the next input slot and dispatch.
        False if the ring is full (drain with yuv_acquire first)."""
        p = _LIB.r3dm_yuv_input(self.id)
        if not p:
            return False
        view = np.ctypeslib.as_array(
            (ctypes.c_uint8 * (self.width * self.height * 4)).from_address(p))
        view.reshape((self.height, self.width, 4))[...] = frame
        if _LIB.r3dm_yuv_submit(self.id) != 0:
            raise RuntimeError("r3dm_yuv_submit failed")
        return True

    def yuv_acquire(self, force=False):
        """Oldest finished conversion as planar yuv420p bytes, or None while the
        pipeline fills. COPIED: the slot is reused `ring` frames later and the
        writer queue is deeper than the ring."""
        p = _LIB.r3dm_yuv_acquire(self.id, 1 if force else 0)
        if not p:
            return None
        return bytes(np.ctypeslib.as_array(
            (ctypes.c_uint8 * self._ylen).from_address(p)))

    def yuv_from_ring(self):
        """Convert the oldest finished render-target slot straight to yuv420p.
        No readback, no host RGBA copy -- 3.11 MB crosses instead of 19.7 MB.
        Valid ONLY when nothing is drawn on the CPU after the pass."""
        p = _LIB.r3dm_yuv_from_ring(self.id)
        if not p:
            return None
        return bytes(np.ctypeslib.as_array(
            (ctypes.c_uint8 * self._ylen).from_address(p)))

    def commit_yuv(self):
        """Commit the open pass with the yuv conversion chained into the SAME
        command buffer, and return WITHOUT waiting."""
        if _LIB.r3dm_commit_yuv(self.id) != 0:
            raise RuntimeError("r3dm_commit_yuv failed")

    def yuv_ring_acquire(self, depth=3, force=False):
        """Oldest finished yuv420p frame, or None while the pipeline fills."""
        p = _LIB.r3dm_yuv_ring_acquire(self.id, int(depth), 1 if force else 0)
        if not p:
            return None
        return bytes(np.ctypeslib.as_array(
            (ctypes.c_uint8 * self._ylen).from_address(p)))

    # ---- GPU flashlight (FL mod vignette) ----------------------------------
    def flash_inline(self, params):
        """Multiply the OPEN pass by the flashlight vignette. Call between the
        playfield sprites and the in-pass HUD -- that is where the CPU pass sat.
        6 floats: w, h, cx, cy, R, Ro."""
        if not hasattr(self, "_fparams"):
            self._fparams = (ctypes.c_float * 6)()
        for i, v in enumerate(params):
            self._fparams[i] = float(v)
        if _LIB.r3dm_flash_inline(self.id, self._fparams) != 0:
            raise RuntimeError("r3dm_flash_inline failed")

    # ---- GPU health bar ----------------------------------------------------
    def hp_init(self, sdfs):
        """sdfs: [main.d_perp, main.t_near, glow.d_perp, glow.t_near]."""
        if _LIB.r3dm_hp_init(self.id) != 0:
            raise RuntimeError("r3dm_hp_init failed")
        for i, a in enumerate(sdfs):
            f = np.ascontiguousarray(a, dtype=np.float32)
            if _LIB.r3dm_hp_sdf(self.id, i, f.shape[1], f.shape[0],
                                f.ctypes.data_as(ctypes.c_void_p)) != 0:
                raise RuntimeError(f"r3dm_hp_sdf({i}) failed")
        self._hsrc = _LIB.r3dm_hp_src(self.id)
        if not self._hsrc:
            raise RuntimeError("r3dm_hp_src returned NULL")
        self._hparams = (ctypes.c_float * 36)()

    def hp_src_view(self):
        return self._wrap(self._hsrc)

    def hp_run(self, params):
        for i, v in enumerate(params):
            self._hparams[i] = float(v)
        p = _LIB.r3dm_hp_run(self.id, ctypes.byref(self._hparams))
        if not p:
            raise RuntimeError("r3dm_hp_run failed")
        # CONTRACT: this is a view of ONE reused destination buffer, valid only
        # until the next hp_run. Its single caller (_gpu_hp_bar) composites it
        # into the frame before returning, so that holds. Do NOT queue it --
        # death_run had exactly this shape and corrupted queued frames.
        return self._wrap(p)

    def hp_inline(self, params):
        """Draw the bar into the OPEN sprite pass -- no round trip. Call between
        begin() and commit(), after the scene sprites."""
        for i, v in enumerate(params):
            self._hparams[i] = float(v)
        if _LIB.r3dm_hp_inline(self.id, ctypes.byref(self._hparams)) != 0:
            raise RuntimeError("r3dm_hp_inline failed")

    def pill_inline(self, params):
        """One src-over rounded pill into the open pass (11 floats)."""
        if not hasattr(self, "_pparams"):
            self._pparams = (ctypes.c_float * 11)()
        for i, v in enumerate(params):
            self._pparams[i] = float(v)
        if _LIB.r3dm_pill_inline(self.id, ctypes.byref(self._pparams)) != 0:
            raise RuntimeError("r3dm_pill_inline failed")

    def close(self):
        if getattr(self, "id", -1) >= 0:
            _LIB.r3dm_destroy(self.id); self.id = -1
    def __del__(self):
        try: self.close()
        except Exception: pass
