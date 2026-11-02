"""H.264 MP4 writer settings of the Record button -- Qt-free, so the frozen app's ``--self-test`` can use them.

``mrstudio.gui.recorder`` streams its frames through :func:`open_writer`; ``packaging/launcher.py --self-test``
writes a short clip with the same settings, so a bundle whose ffmpeg cannot encode H.264 fails the build
instead of the user's first recording.
"""
from __future__ import annotations

# imageio-ffmpeg writer options: H.264 in yuv420p (plays everywhere), frame sizes rounded to a multiple of 2 (not
# imageio's default 16), quiet ffmpeg, a fast preset tuned for flat-shaded 3-D views.
WRITER_KWARGS = {
    "codec": "libx264",
    "quality": 8,
    "pixelformat": "yuv420p",
    "macro_block_size": 2,
    "ffmpeg_log_level": "error",
    "ffmpeg_params": ["-preset", "veryfast", "-tune", "animation"],
}


def open_writer(path: str, fps: float):
    """An imageio writer for an H.264 MP4 at ``path`` (``append_data(rgb_uint8_array)``, then ``close()``)."""
    import imageio.v2 as iio
    kwargs = dict(WRITER_KWARGS, ffmpeg_params=list(WRITER_KWARGS["ffmpeg_params"]))
    return iio.get_writer(path, fps=fps, **kwargs)
