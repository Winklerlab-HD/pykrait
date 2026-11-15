import sys
import warnings

import numpy as np
import pytest

pyczi = pytest.importorskip("pylibCZIrw.czi")

import pykrait.io.images as images  # noqa: E402
from pykrait.io.images import load_timelapse_lazy  # noqa: E402


@pytest.fixture(scope="module")
def czi_video(tmp_path_factory):
    """Two-channel CZI with distinct values in every plane."""
    path = tmp_path_factory.mktemp("czi") / "video.czi"
    rng = np.random.default_rng(0)
    video = rng.integers(0, 4000, size=(20, 2, 48, 40), dtype=np.uint16)
    with pyczi.create_czi(str(path), exist_ok=True) as czi:
        for t in range(video.shape[0]):
            for c in range(video.shape[1]):
                czi.write(data=video[t, c][..., None], plane={"T": t, "C": c})
    return str(path), video


def _reads_directly(data) -> bool:
    return any(layer.startswith("czi-planes") for layer in data.dask.layers)


def _load(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the synthetic file has no frame interval
        return load_timelapse_lazy(path)[0]


def test_direct_reader_is_used_and_matches_bioio(czi_video, monkeypatch):
    path, video = czi_video
    direct = _load(path)
    assert _reads_directly(direct)

    monkeypatch.setattr(images, "_czi_dask_array", lambda *args, **kwargs: None)
    bioio = _load(path)
    assert direct.shape == bioio.shape == video.shape
    assert direct.dtype == bioio.dtype
    np.testing.assert_array_equal(np.asarray(direct), np.asarray(bioio))
    np.testing.assert_array_equal(np.asarray(direct), video)


def test_direct_reader_supports_the_slices_of_the_analysis(czi_video):
    path, video = czi_video
    data = _load(path)
    np.testing.assert_array_equal(np.asarray(data[3:11]), video[3:11])
    np.testing.assert_array_equal(np.asarray(data[:, 1]), video[:, 1])
    np.testing.assert_array_equal(np.asarray(data[7, :, 5:9, 2]), video[7, :, 5:9, 2])


def test_falls_back_to_bioio_without_pylibczirw(czi_video, monkeypatch):
    path, video = czi_video
    monkeypatch.setitem(sys.modules, "pylibCZIrw", None)
    data = _load(path)
    assert not _reads_directly(data)
    np.testing.assert_array_equal(np.asarray(data), video)


def test_falls_back_to_bioio_if_the_frames_differ(czi_video, monkeypatch):
    path, video = czi_video
    original = images._CziPlanes.__getitem__
    monkeypatch.setattr(
        images._CziPlanes, "__getitem__", lambda self, key: original(self, key) + 1
    )
    data = _load(path)
    assert not _reads_directly(data)
    np.testing.assert_array_equal(np.asarray(data), video)
