import os
from collections import namedtuple

import dask.array as da
import numpy as np
import pytest

import pykrait.io.frame_cache as frame_cache
from pykrait.io.frame_cache import FrameCache, choose_video_cache, iter_frame_blocks

GB = 1024**3


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """Sets the available memory, the free disk space and whether the video is on another drive."""
    state = {"available": 8 * GB, "free": 100 * GB, "same_drive": False}
    memory = namedtuple("memory", "available")
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        frame_cache.psutil, "virtual_memory", lambda: memory(state["available"])
    )
    monkeypatch.setattr(
        frame_cache.shutil, "disk_usage", lambda path: usage(0, 0, state["free"])
    )
    real_stat = os.stat
    video = tmp_path / "video.czi"
    video.write_bytes(b"")

    def stat(path, *args, **kwargs):
        result = real_stat(path, *args, **kwargs)
        if str(path) == str(video) and not state["same_drive"]:
            return os.stat_result(result[:2] + (result.st_dev + 1,) + result[3:])
        return result

    monkeypatch.setattr(frame_cache.os, "stat", stat)
    state["video"] = str(video)
    return state


def test_frames_are_kept_in_memory_if_they_need_at_most_half_of_it(machine):
    assert choose_video_cache(4 * GB, machine["video"])[0] == "memory"
    assert choose_video_cache(5 * GB, machine["video"])[0] == "disk"


def test_frames_go_to_disk_only_for_videos_on_another_drive(machine):
    machine["same_drive"] = True
    location, reason = choose_video_cache(5 * GB, machine["video"])
    assert location == "none" and "drive" in reason


def test_disk_cache_keeps_five_gb_free(machine):
    machine["free"] = 9.9 * GB
    location, reason = choose_video_cache(5 * GB, machine["video"])
    assert location == "none" and "free" in reason
    assert choose_video_cache(5 * GB, machine["video"], mode="disk")[0] == "none"


def test_forced_modes(machine):
    assert choose_video_cache(50 * GB, machine["video"], mode="memory")[0] == "memory"
    assert choose_video_cache(1, machine["video"], mode="disk")[0] == "disk"
    assert choose_video_cache(1, machine["video"], mode="none")[0] == "none"
    with pytest.raises(ValueError):
        choose_video_cache(1, machine["video"], mode="ram")


def test_no_temporary_file_without_save_temp_file(machine):
    # frames that fit into memory are still kept there
    assert choose_video_cache(4 * GB, machine["video"], save_temp_file=False)[0] == (
        "memory"
    )
    location, reason = choose_video_cache(
        5 * GB, machine["video"], save_temp_file=False
    )
    assert location == "none" and "temporary file" in reason
    with pytest.raises(ValueError, match="save_temp_file"):
        choose_video_cache(1, machine["video"], mode="disk", save_temp_file=False)


@pytest.mark.parametrize("prefetch", [True, False])
def test_iter_frame_blocks_returns_every_frame_in_order(prefetch):
    video = np.arange(10 * 3 * 4).reshape(10, 3, 4)
    blocks = list(
        iter_frame_blocks(da.from_array(video, chunks=2), 4, prefetch=prefetch)
    )
    assert [start for start, _ in blocks] == [0, 4, 8]
    np.testing.assert_array_equal(np.concatenate([block for _, block in blocks]), video)


@pytest.mark.parametrize("location", ["memory", "disk"])
def test_frame_cache_serves_the_kept_channel(location):
    rng = np.random.default_rng(0)
    video = rng.integers(0, 1000, size=(9, 3, 5, 6), dtype=np.uint16)
    with FrameCache(video.shape, video.dtype, channel=1, location=location) as cache:
        path = cache.path
        assert (path is not None) == (location == "disk")
        for start in range(0, 9, 4):
            cache.store(start, video[start : start + 4])
        timelapse = cache.as_timelapse()
        assert timelapse.shape == video.shape
        np.testing.assert_array_equal(np.asarray(timelapse[:, 1]), video[:, 1])
        np.testing.assert_array_equal(
            np.asarray(timelapse[2:7, 1, 1:3]), video[2:7, 1, 1:3]
        )
        np.testing.assert_array_equal(np.asarray(timelapse[4, 1:2]), video[4, 1:2])
        with pytest.raises(IndexError):
            np.asarray(timelapse[:, 0])
    if path is not None:
        assert not os.path.exists(path)


def test_frame_cache_falls_back_if_it_cannot_be_created(monkeypatch):
    monkeypatch.setattr(
        frame_cache.tempfile,
        "mkstemp",
        lambda **kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    video = da.zeros((4, 1, 8, 8), dtype=np.uint16)
    assert FrameCache.for_timelapse(video, channel=0, mode="disk") is None
