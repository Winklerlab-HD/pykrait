"""
Reading a timelapse block by block, and keeping the frames of one channel after the first
pass so that a second pass does not have to read the video file again.

The segmentation reads the whole video for the projection, and the intensity extraction
reads it again once the masks exist. On slow drives every pass is limited by the drive,
and operating systems do not reliably keep gigabyte-sized files of external drives in
their cache. A FrameCache keeps the frames the intensity extraction needs, either in
memory or, if they do not fit, in a temporary file on another (usually the internal)
drive.
"""

import logging
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, Literal, Tuple

import dask.array as da
import numpy as np
import psutil

logger = logging.getLogger(__name__)

# the frames are kept in memory if they need at most this fraction of the available
# memory, the rest is left for the segmentation and the analysis
MEMORY_FRACTION = 0.5
# disk space that has to stay free on the drive of the temporary file
MIN_FREE_DISK_SPACE = 5 * 1024**3

VideoCacheMode = Literal["auto", "memory", "disk", "none"]


def iter_frame_blocks(
    timelapse, block_size: int, prefetch: bool = True
) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Reads a timelapse block by block along its first axis. With prefetch, the next block
    is read on a background thread while the current one is processed, so reading and
    computing overlap.

    :param timelapse: lazily loaded array, e.g. a Dask array of shape (T, ...)
    :param block_size: number of frames per block
    :type block_size: int
    :param prefetch: whether to read the next block in the background, defaults to True
    :type prefetch: bool, optional
    :yield: the index of the first frame and the frames of each block as a numpy array
    :rtype: Iterator[Tuple[int, np.ndarray]]
    """
    starts = range(0, timelapse.shape[0], block_size)

    def read(start: int) -> np.ndarray:
        return np.asarray(timelapse[start : start + block_size])

    if not prefetch or len(starts) < 2:
        for start in starts:
            yield start, read(start)
        return

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pykrait-read") as pool:
        pending = pool.submit(read, starts[0])
        for index, start in enumerate(starts):
            block = pending.result()
            if index + 1 < len(starts):
                pending = pool.submit(read, starts[index + 1])
            yield start, block


def choose_video_cache(
    n_bytes: int,
    video_path: str = None,
    mode: VideoCacheMode = "auto",
    save_temp_file: bool = True,
) -> Tuple[str, str]:
    """
    Decides where to keep the frames of one channel between the two passes over a video.

    In "auto" mode the frames are kept in memory if they need at most half of the
    currently available memory. Otherwise, if the video lies on a different drive than
    the temporary directory and that drive keeps at least 5 GB free after writing the
    frames, they are kept in a temporary file there. Otherwise they are not kept and the
    video is read again. With `save_temp_file` set to False, no temporary file is written
    and frames that do not fit into memory are read again.

    :param n_bytes: size of the frames to keep in bytes
    :type n_bytes: int
    :param video_path: path of the video, used to compare its drive with the temporary directory's, defaults to None
    :type video_path: str, optional
    :param mode: "auto", or "memory", "disk" or "none" to force a location, defaults to "auto"
    :type mode: VideoCacheMode, optional
    :param save_temp_file: whether "auto" mode may keep the frames in a temporary file on disk, defaults to True
    :type save_temp_file: bool, optional
    :raises ValueError: if the mode is unknown, or "disk" is requested without allowing a temporary file
    :return: returns the location ("memory", "disk" or "none") and the reason for the choice
    :rtype: Tuple[str, str]
    """
    if mode not in ("auto", "memory", "disk", "none"):
        raise ValueError(
            f"Unknown video cache mode '{mode}', expected 'auto', 'memory', 'disk' or 'none'."
        )
    if mode == "disk" and not save_temp_file:
        raise ValueError(
            "The video cache mode 'disk' needs a temporary file, but save_temp_file is False."
        )
    if mode in ("memory", "none"):
        return mode, "requested"

    size = _format_bytes(n_bytes)
    if mode == "auto":
        available = psutil.virtual_memory().available
        if n_bytes <= MEMORY_FRACTION * available:
            return "memory", f"{size} of {_format_bytes(available)} available memory"

    if not save_temp_file:
        return "none", f"{size} do not fit into memory, and no temporary file is saved"

    directory = tempfile.gettempdir()
    free = shutil.disk_usage(directory).free
    if free - n_bytes < MIN_FREE_DISK_SPACE:
        return "none", (
            f"{size} do not fit into memory, and only {_format_bytes(free)} are free in {directory}"
        )
    if mode == "disk":
        return "disk", "requested"
    if (
        video_path is not None
        and os.stat(video_path).st_dev == os.stat(directory).st_dev
    ):
        # reading the video again is about as fast as reading a copy on the same drive
        return (
            "none",
            f"{size} do not fit into memory, and the video is on the drive of {directory}",
        )
    return (
        "disk",
        f"{size} do not fit into memory, {_format_bytes(free)} free in {directory}",
    )


def _format_bytes(n_bytes: float) -> str:
    return f"{n_bytes / 1024**3:.1f} GB"


class FrameCache:
    """
    Keeps the frames of one channel of a timelapse, in memory or in a temporary file.

    Use it as a context manager, so the temporary file is deleted afterwards.

    :param shape: (T, C, Y, X) shape of the timelapse
    :type shape: tuple
    :param dtype: dtype of the timelapse
    :param channel: index of the channel to keep
    :type channel: int
    :param location: "memory" or "disk"
    :type location: str
    """

    def __init__(self, shape: tuple, dtype, channel: int, location: str):
        if location not in ("memory", "disk"):
            raise ValueError(f"Unknown frame cache location '{location}'.")
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.channel = channel
        self.location = location
        self.path = None
        frames_shape = (self.shape[0],) + self.shape[2:]
        if location == "memory":
            self.frames = np.empty(frames_shape, dtype=self.dtype)
        else:
            handle, self.path = tempfile.mkstemp(
                prefix="pykrait_frames_", suffix=".npy"
            )
            os.close(handle)
            self.frames = np.lib.format.open_memmap(
                self.path, mode="w+", dtype=self.dtype, shape=frames_shape
            )

    @classmethod
    def for_timelapse(
        cls,
        timelapse,
        channel: int,
        video_path: str = None,
        mode: VideoCacheMode = "auto",
        save_temp_file: bool = True,
    ) -> "FrameCache":
        """
        Creates a cache for one channel of a timelapse, where choose_video_cache decides.

        :return: returns the cache, or None if the frames are not kept
        :rtype: FrameCache
        """
        return cls.for_shape(
            timelapse.shape,
            timelapse.dtype,
            channel,
            video_path=video_path,
            mode=mode,
            save_temp_file=save_temp_file,
        )

    @classmethod
    def for_shape(
        cls,
        shape: tuple,
        dtype,
        channel: int,
        video_path: str = None,
        mode: VideoCacheMode = "auto",
        save_temp_file: bool = True,
    ) -> "FrameCache":
        """
        Creates a cache for one channel of a (T, C, Y, X) video that may not exist yet,
        e.g. a motion-corrected video whose frames are stored while they are corrected.

        :return: returns the cache, or None if the frames are not kept
        :rtype: FrameCache
        """
        n_frames, _, height, width = shape
        n_bytes = n_frames * height * width * np.dtype(dtype).itemsize
        location, reason = choose_video_cache(n_bytes, video_path, mode, save_temp_file)
        if location == "none":
            logger.info(f"Reading the video again for the intensities: {reason}.")
            return None
        where = "in memory" if location == "memory" else "in a temporary file"
        try:
            cache = cls(shape, dtype, channel, location)
        except Exception as e:
            # the cache only saves time, so the video is read again instead
            logger.warning(
                f"Could not keep the intensity channel {where}, reading the video again: {type(e).__name__}: {e}"
            )
            return None
        logger.info(f"Keeping the intensity channel {where} ({reason}).")
        return cache

    def store(self, start: int, block: np.ndarray) -> None:
        """Stores the cached channel of a block of (n, C, Y, X) frames starting at `start`."""
        self.frames[start : start + block.shape[0]] = block[:, self.channel]

    def as_timelapse(self) -> da.Array:
        """
        Returns the cached frames as a Dask array of the full (T, C, Y, X) shape, so it can
        be used in place of the timelapse. Only the cached channel can be read.
        """
        return da.from_array(
            _CachedChannel(self),
            chunks=(max(1, self.shape[0]), 1) + self.shape[2:],
            name=f"pykrait-frame-cache-{id(self)}",
            asarray=True,
            meta=np.empty((0,) * 4, dtype=self.dtype),
        )

    def close(self) -> None:
        """Releases the frames and deletes the temporary file."""
        self.frames = None
        if self.path is not None and os.path.exists(self.path):
            os.remove(self.path)

    def __enter__(self) -> "FrameCache":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class _CachedChannel:
    """Array-like (T, C, Y, X) view of a FrameCache that only serves the cached channel."""

    def __init__(self, cache: FrameCache):
        self._cache = cache
        self.shape = cache.shape
        self.dtype = cache.dtype
        self.ndim = 4

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))
        channels = np.arange(self.shape[1])[key[1]]
        if np.any(np.atleast_1d(channels) != self._cache.channel):
            raise IndexError(
                f"Only channel {self._cache.channel} was kept, channels {np.atleast_1d(channels).tolist()} were requested."
            )
        frames = self._cache.frames[key[0]][..., key[2], key[3]]
        if np.ndim(channels) == 0:
            return frames  # the channel axis was indexed away
        # put the channel axis back, after the time axis unless that was indexed away
        time_axis_kept = np.ndim(np.arange(self.shape[0])[key[0]]) == 1
        return np.expand_dims(frames, axis=1 if time_axis_kept else 0)
