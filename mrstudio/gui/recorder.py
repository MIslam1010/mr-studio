"""Screen recording of the 3-D view or the whole window -> MP4 (H.264) or GIF, plus PNG screenshots.

Frames are grabbed on the GUI thread by a QTimer (``QOpenGLWidget.grabFramebuffer`` for a 3-D view,
which includes the GL text labels, or ``QWidget.grab`` for the window) and handed through a bounded
queue to a worker thread that scales them and streams them into a temporary H.264 MP4 (imageio-ffmpeg).
Memory stays constant whatever the length. On save the temp MP4 is moved (MP4) or converted to an
optimised GIF (Pillow, shared palette, size cap) -- again in a worker thread.

    rec = Recorder()
    rec.start(view.grab_frame, fps=30)      # any callable returning a QImage / QPixmap
    ...
    tmp = rec.stop()                        # waits for the encoder to flush, returns temp .mp4
    rec.export(tmp, "/path/clip.gif", block=True)
"""
from __future__ import annotations

import os
import queue
import shutil
import tempfile
import threading
import time

import numpy as np
from PySide6 import QtCore, QtGui

MAX_VIDEO_WIDTH = 1920   # Retina grabs are 2x: cap so files stay reasonable
GIF_MAX_WIDTH = 720
GIF_FPS = 15
GIF_COLORS = 128


def qimage_to_rgb(img: QtGui.QImage, max_width: int | None = MAX_VIDEO_WIDTH,
                  size: tuple[int, int] | None = None) -> np.ndarray:
    """QImage -> HxWx3 uint8 (optionally scaled to ``size`` = (w, h) or capped to ``max_width``)."""
    if isinstance(img, QtGui.QPixmap):
        img = img.toImage()
    if size is not None and (img.width(), img.height()) != size:
        img = img.scaled(size[0], size[1], QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
    elif max_width and img.width() > max_width:
        img = img.scaledToWidth(max_width, QtCore.Qt.SmoothTransformation)
    img = img.convertToFormat(QtGui.QImage.Format_RGB888)
    w, h, bpl = img.width(), img.height(), img.bytesPerLine()
    arr = np.frombuffer(img.constBits(), np.uint8, count=bpl * h).reshape(h, bpl)
    return arr[:, : 3 * w].reshape(h, w, 3).copy()


def _even(a: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    return a[: h - h % 2, : w - w % 2]


class Recorder(QtCore.QObject):
    """Grab frames at a fixed rate and encode them in a background thread."""
    tick = QtCore.Signal(float)            # elapsed seconds while recording
    exported = QtCore.Signal(str)          # output path after export()
    failed = QtCore.Signal(str)            # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._timer = QtCore.QTimer(self)
        self._timer.setTimerType(QtCore.Qt.PreciseTimer)
        self._timer.timeout.connect(self._grab)
        self._grab_fn = None
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._error: str | None = None
        self.fps = 30
        self.tmp_path: str | None = None
        self.frames = 0
        self.dropped = 0
        self._t0 = 0.0
        self._size = None

    # ------------------------------------------------------------------ recording
    @property
    def recording(self) -> bool:
        return self._timer.isActive()

    def elapsed(self) -> float:
        return time.perf_counter() - self._t0 if self.recording else 0.0

    def start(self, grab_fn, fps: int = 30):
        if self.recording:
            return
        import imageio.v2 as iio  # noqa: F401  (fail early if the dependency is missing)
        self._grab_fn, self.fps = grab_fn, int(fps)
        fd, self.tmp_path = tempfile.mkstemp(prefix="mrstudio_rec_", suffix=".mp4")
        os.close(fd)
        self._queue = queue.Queue(maxsize=90)   # ~3 s of buffered frames
        self._error, self.frames, self.dropped, self._size = None, 0, 0, None
        self._thread = threading.Thread(target=self._encode_loop, args=(self.tmp_path, self.fps), daemon=True)
        self._thread.start()
        self._t0 = time.perf_counter()
        self._timer.start(max(1, int(round(1000 / self.fps))))
        self._grab()

    def _grab(self):
        try:
            img = self._grab_fn()
            if isinstance(img, QtGui.QPixmap):
                img = img.toImage()
            if img is None or img.isNull():
                return
            if self._size is None:  # fix the output size on the first frame (H.264 needs even sizes)
                w, h = img.width(), img.height()
                if w > MAX_VIDEO_WIDTH:
                    h, w = int(round(h * MAX_VIDEO_WIDTH / w)), MAX_VIDEO_WIDTH
                self._size = (w - w % 2, h - h % 2)
            # keep real-time playback: duplicate the frame if grabbing fell behind the clock
            due = int((time.perf_counter() - self._t0) * self.fps) + 1
            n = min(max(due - self.frames, 1), 4)
            try:
                self._queue.put_nowait((img.copy(), n))
                self.frames += n
            except queue.Full:
                self.dropped += 1
            self.tick.emit(time.perf_counter() - self._t0)
        except Exception as e:  # pragma: no cover
            self._error = str(e)

    def _encode_loop(self, path, fps):
        import imageio.v2 as iio
        writer = None
        try:
            writer = iio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                                    macro_block_size=2, ffmpeg_log_level="error",
                                    ffmpeg_params=["-preset", "veryfast", "-tune", "animation"])
            while True:
                item = self._queue.get()
                if item is None:
                    break
                img, n = item
                arr = _even(qimage_to_rgb(img, size=self._size))
                for _ in range(n):
                    writer.append_data(arr)
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            while True:  # drain so the GUI never blocks on a full queue
                try:
                    if self._queue.get_nowait() is None:
                        break
                except queue.Empty:
                    time.sleep(0.01)
        finally:
            if writer is not None:
                try:
                    writer.close()
                except Exception as e:  # pragma: no cover
                    self._error = self._error or str(e)

    def stop(self) -> str | None:
        """Stop grabbing, wait for the encoder to flush; returns the temp MP4 path (None on error)."""
        if self._thread is None:
            return None
        self._timer.stop()
        self._queue.put(None)
        self._thread.join(timeout=60)
        self._thread = None
        if self._error or self.frames == 0:
            self.failed.emit(self._error or "no frames were recorded")
            return None
        return self.tmp_path

    def cancel(self):
        p = self.stop()
        if p and os.path.exists(p):
            os.remove(p)

    # ------------------------------------------------------------------ export
    def export(self, tmp_mp4: str, out_path: str, block: bool = False):
        """Move the temp MP4 to ``out_path`` (.mp4) or convert it to an optimised GIF (.gif)."""
        def job():
            try:
                if out_path.lower().endswith(".gif"):
                    mp4_to_gif(tmp_mp4, out_path)
                    os.remove(tmp_mp4)
                else:
                    shutil.move(tmp_mp4, out_path)
                self.exported.emit(out_path)
            except Exception as e:
                self.failed.emit(f"{type(e).__name__}: {e}")
        if block:
            job()
        else:
            threading.Thread(target=job, daemon=True).start()


def mp4_to_gif(src: str, dst: str, max_width: int = GIF_MAX_WIDTH, fps: int = GIF_FPS,
               colors: int = GIF_COLORS):
    """Convert a video to a small looping GIF: <= max_width px, ~fps, one shared adaptive palette."""
    import imageio.v2 as iio
    from PIL import Image

    rd = iio.get_reader(src)
    src_fps = float(rd.get_meta_data().get("fps", 30)) or 30.0
    step = max(1, int(round(src_fps / fps)))
    frames = []
    for i, fr in enumerate(rd):
        if i % step:
            continue
        im = Image.fromarray(fr)
        if im.width > max_width:
            im = im.resize((max_width, int(round(im.height * max_width / im.width))), Image.LANCZOS)
        frames.append(im)
    rd.close()
    if not frames:
        raise RuntimeError("video has no frames")
    # shared palette from a few sample frames -> no palette flicker, better inter-frame compression
    idx = np.linspace(0, len(frames) - 1, min(len(frames), 6)).astype(int)
    strip = Image.new("RGB", (frames[0].width, frames[0].height * len(idx)))
    for k, j in enumerate(idx):
        strip.paste(frames[j], (0, k * frames[0].height))
    pal = strip.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(dst, save_all=True, append_images=q[1:], duration=int(round(1000 * step / src_fps)),
              loop=0, optimize=True, disposal=1)
