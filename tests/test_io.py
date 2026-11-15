import pytest
import os
import datetime
import tempfile
import numpy as np
import pandas as pd
import dask.array as da
from unittest.mock import patch, MagicMock
from pykrait.io.images import load_timelapse_lazy, read_label_image, _reorder_dask_array
from pykrait.io.files import (
    get_files_from_folder,
    get_pykrait_version,
    save_Txnrois,
    read_Txnrois,
)
from pathlib import Path


def test_invalid_file_path_type():
    with pytest.raises(TypeError):
        load_timelapse_lazy(123)  # Non-string path


def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_timelapse_lazy("nonexistent_file.tif")


def test_invalid_extension(tmp_path):
    file = tmp_path / "test.invalid"
    file.write_text("fake")
    with pytest.raises(ValueError):
        load_timelapse_lazy(str(file))


@patch("pykrait.io.images.BioImage")
@patch("os.path.exists", return_value=True)
def test_valid_single_z_image(mock_exists, mock_aicsimage):
    mock_data = da.from_array(np.zeros((3, 2, 1, 128, 128)), chunks=(1, 1, 1, 128, 128))

    mock_instance = MagicMock()
    mock_instance.dask_data = mock_data

    mock_instance.physical_pixel_sizes.Y = 0.5
    mock_instance.physical_pixel_sizes.X = 0.6
    mock_instance.metadata.find.return_value = None  # to fail regex load of metadata
    mock_instance.time_interval = datetime.timedelta(seconds=4)

    mock_aicsimage.return_value = mock_instance

    dask_array, frame_interval, y_um, x_um = load_timelapse_lazy("file.czi")

    assert isinstance(dask_array, da.Array)
    assert dask_array.shape == (3, 2, 128, 128)
    assert frame_interval == 4.0
    assert y_um == 0.5
    assert x_um == 0.6


@patch("pykrait.io.images.BioImage")
@patch("os.path.exists", return_value=True)
def test_missing_metadata(mock_exists, mock_aicsimage):
    mock_data = da.from_array(np.zeros((100, 1, 1, 64, 64)), chunks=(100, 1, 1, 64, 64))

    mock_instance = MagicMock()
    mock_instance.dask_data = mock_data

    # No metadata available
    mock_instance.physical_pixel_sizes = None
    mock_instance.physical_size_t = None
    mock_instance.metadata = None
    mock_aicsimage.return_value = mock_instance

    # Expecting a warning about missing metadata
    with pytest.warns(
        UserWarning, match="Failed to extract frame interval from metadata:"
    ):
        dask_array, frame_interval, y_um, x_um = load_timelapse_lazy("file.tif")

    assert isinstance(dask_array, da.Array)
    assert dask_array.shape == (100, 1, 64, 64)
    assert frame_interval is None
    assert y_um is None
    assert x_um is None


@patch("pykrait.io.images.BioImage")
@patch("os.path.exists", return_value=True)
def test_invalid_dimensions(mock_exists, mock_aicsimage):
    # Not a 5D array
    mock_data = da.from_array(np.zeros((3, 2, 128, 128)), chunks=(1, 1, 128, 128))

    mock_instance = MagicMock()
    mock_instance.dask_data = mock_data
    mock_instance.metadata = "/nfinterval=4.000/n"
    mock_aicsimage.return_value = mock_instance

    with pytest.raises(ValueError, match="Expected 5D image"):
        load_timelapse_lazy("file.tiff")


@patch("pykrait.io.images.BioImage")
@patch("os.path.exists", return_value=True)
def test_unhandled_z_dimension(mock_exists, mock_aicsimage):
    # Z > 1 should raise error
    mock_data = da.from_array(np.zeros((5, 1, 3, 64, 64)), chunks=(1, 1, 1, 64, 64))

    mock_instance = MagicMock()
    mock_instance.dask_data = mock_data
    mock_instance.metadata = "/nfinterval=4.000/n"
    mock_aicsimage.return_value = mock_instance

    with pytest.raises(ValueError, match="Cannot drop Z dimension automatically"):
        load_timelapse_lazy("file.tiff")


IO_TEST_DATA = Path(__file__).parent / "io_test_data"


def test_real_tif_matches_expected():
    test_file = str(IO_TEST_DATA / "iotestdata1.tif")
    assert os.path.exists(test_file), f"Test file missing: {test_file}"

    # Load actual result
    dask_arr, frame_interval, y_um, x_um = load_timelapse_lazy(test_file)
    actual_array = dask_arr.compute()

    # Load saved expected results
    expected_array = np.load(IO_TEST_DATA / "expected_array_iotestdata1.npy")
    expected_metadata = np.load(IO_TEST_DATA / "expected_metadata_iotestdata1.npy")

    # Compare image arrays
    assert actual_array.shape == expected_array.shape
    assert np.allclose(actual_array, expected_array, atol=1e-5), "Image array mismatch"

    # Compare metadata (frame_interval, y_um, x_um)
    actual_metadata = np.array(
        [
            frame_interval if frame_interval is not None else -1,
            y_um if y_um is not None else -1,
            x_um if x_um is not None else -1,
        ]
    )
    assert np.allclose(actual_metadata, expected_metadata, atol=1e-2), (
        "Metadata mismatch"
    )  # the expected metadata was saved rounded to two decimal places


@patch("pykrait.io.images.BioImage")
@patch("os.path.exists", return_value=True)
def test_three_non_singleton_dims(mock_exists, mock_bioimage):
    shape = (1, 10, 1, 128, 128)
    dask_data = da.random.random(shape, chunks=(1, 1, 1, 128, 128))

    mock_instance = MagicMock()
    mock_instance.dask_data = dask_data
    mock_instance.physical_pixel_sizes = MagicMock(Y=1.0, X=1.0)
    mock_instance.metadata.find.return_value = None  # to fail regex load of metadata
    mock_instance.time_interval = datetime.timedelta(seconds=2)
    mock_bioimage.return_value = mock_instance

    result, frame_interval, y_um, x_um = load_timelapse_lazy("mock_file.czi")

    assert isinstance(result, da.Array)
    assert result.shape == (10, 1, 128, 128)
    assert frame_interval == 2.0


@patch("pykrait.io.images.os.path.exists", return_value=True)
@patch("pykrait.io.images.BioImage")
def test_read_label_image_returns_2d(mock_bioimage, mock_exists):
    mock_array = np.random.randint(0, 5, size=(1, 1, 1, 64, 64))
    mock_instance = MagicMock()
    mock_instance.dask_data.compute.return_value = mock_array
    mock_bioimage.return_value = mock_instance

    result = read_label_image("dummy_path.tiff")
    assert isinstance(result, np.ndarray)
    assert result.shape == (64, 64)
    assert result.ndim == 2


def test_get_files_from_folder(tmp_path):
    (tmp_path / "file1.czi").write_text("data")
    (tmp_path / "file2.tif").write_text("data")
    (tmp_path / "other.txt").write_text("data")

    results = get_files_from_folder(str(tmp_path), extension=".czi")
    assert len(results) == 1
    assert results[0].endswith("file1.czi")


@pytest.fixture
def experiment_tree(tmp_path):
    """Builds a folder tree with several acquisition types per sample, spread over one
    subfolder per session."""
    filenames = [
        "sample1_treated_timeseries.czi",
        "sample1_timeseries.czi",
        "sample1_timeseries_plane.czi",
        "sample1_snap.czi",
        "sample1_zstack.czi",
        "sample1_overview.czi",
    ]
    for session in ["session_1", "session_2"]:
        session_folder = tmp_path / "sample1" / session
        session_folder.mkdir(parents=True)
        for filename in filenames:
            (session_folder / filename).write_text("data")
    (tmp_path / "sample1" / "positions.xls").write_text("data")
    return tmp_path


def test_get_files_from_folder_is_not_recursive_by_default(experiment_tree):
    assert get_files_from_folder(str(experiment_tree), extension=".czi") == []


def test_get_files_from_folder_recursive_finds_all_subfolders(experiment_tree):
    results = get_files_from_folder(
        str(experiment_tree), extension=".czi", recursive=True
    )
    assert len(results) == 12  # 6 acquisitions in each of the 2 day folders


def test_get_files_from_folder_substring_filter(experiment_tree):
    results = get_files_from_folder(
        str(experiment_tree),
        extension=".czi",
        filter_string="timeseries",
        recursive=True,
    )
    names = {os.path.basename(f) for f in results}
    # a substring also matches the single plane snapshots of the timeseries
    assert names == {
        "sample1_treated_timeseries.czi",
        "sample1_timeseries.czi",
        "sample1_timeseries_plane.czi",
    }


def test_get_files_from_folder_wildcard_filter(experiment_tree):
    results = get_files_from_folder(
        str(experiment_tree),
        extension=".czi",
        filter_string="*_timeseries.czi",
        recursive=True,
    )
    names = {os.path.basename(f) for f in results}
    assert names == {"sample1_treated_timeseries.czi", "sample1_timeseries.czi"}


def test_get_files_from_folder_exclude_string(experiment_tree):
    results = get_files_from_folder(
        str(experiment_tree),
        extension=".czi",
        filter_string="timeseries",
        exclude_string="_plane",
        recursive=True,
    )
    names = {os.path.basename(f) for f in results}
    assert names == {"sample1_treated_timeseries.czi", "sample1_timeseries.czi"}


def test_get_files_from_folder_skips_analysis_folders(experiment_tree):
    analysis_folder = (
        experiment_tree / "sample1" / "session_1" / "Analysis_sample1_timeseries"
    )
    analysis_folder.mkdir()
    (analysis_folder / "sample1_timeseries_motion_corrected.ome.tif").write_text("data")
    (analysis_folder / "sample1_timeseries_tproj.ome.tif").write_text("data")

    results = get_files_from_folder(
        str(experiment_tree), extension=".tif", recursive=True
    )
    assert results == []


def test_get_files_from_folder_is_sorted(experiment_tree):
    results = get_files_from_folder(
        str(experiment_tree), extension=".czi", recursive=True
    )
    assert results == sorted(results)


def test_get_files_from_folder_filter_is_case_sensitive(experiment_tree):
    assert (
        get_files_from_folder(
            str(experiment_tree),
            extension=".czi",
            filter_string="TIMESERIES",
            recursive=True,
        )
        == []
    )


def test_reorder_dask_array_too_few_dims():
    bad_array = da.from_array(np.ones((1, 1, 1, 1, 128)), chunks=(1, 1, 1, 1, 128))
    with pytest.raises(ValueError, match="Expected at least 3 non-1 dimensions"):
        _reorder_dask_array(bad_array)


@patch("pykrait.io.images.os.path.exists", return_value=False)
def test_read_label_image_file_not_found(mock_exists):
    with pytest.raises(FileNotFoundError, match="File not found: .*dummy_path.tiff"):
        read_label_image("dummy_path.tiff")


@patch("pykrait.io.images.os.path.exists", return_value=True)
@patch("pykrait.io.images.BioImage", side_effect=RuntimeError("Mocked failure"))
def test_read_label_image_raises_value_error(mock_bioimage, mock_exists):
    with pytest.raises(
        ValueError, match=r"Failed to read label image.*RuntimeError.*Mocked failure"
    ):
        read_label_image("dummy_path.tiff")


def test_get_pykrait_version_success():
    with patch("pykrait.io.files.version", return_value="1.2.3"):
        assert get_pykrait_version() == "1.2.3"


def test_get_pykrait_version_failure():
    with patch("pykrait.io.files.version", side_effect=Exception("not installed")):
        assert get_pykrait_version() is None


def test_save_peaks_csv_creates_expected_file():
    peaks = np.array([[0.1, 0.2], [0.3, 0.4]])
    frame_interval = 0.5

    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "peaks.csv.zst")
        save_Txnrois(peaks, frame_interval, filepath)

        # Check file was created
        assert os.path.exists(filepath)

        # Read back and validate structure
        df = pd.read_csv(filepath, index_col=0)

        assert list(df.columns) == ["ROI_0", "ROI_1"]
        assert list(df.index.astype(float)) == [0.0, 0.5]
        np.testing.assert_allclose(df.values, peaks)


def test_save_peaks_csv_raises_for_invalid_shape():
    peaks = np.random.rand(10)  # 1D array, invalid
    with pytest.raises(ValueError, match="Input array must be 2D"):
        save_Txnrois(peaks, frame_interval=1.0, filepath="dummy.csv")


def test_save_peaks_csv_raises_for_invalid_type():
    peaks = "not an array"  # Invalid type
    with pytest.raises(TypeError, match="Input must be a numpy ndarray"):
        save_Txnrois(peaks, frame_interval=1.0, filepath="dummy.csv")


def test_read_Txnrois_happy_path_roundtrip(tmp_path: Path):
    T, R = 5, 3
    df = pd.DataFrame(
        {
            "roi_0": [0.0, 1.0, 2.0, 3.0, 4.0],
            "roi_1": [10, 11, 12, 13, 14],
            "roi_2": np.linspace(0.5, 2.5, T),
        },
        index=pd.Index(range(T), name="time"),
    )
    csv_path = tmp_path / "traces.csv"
    df.to_csv(csv_path)

    arr = read_Txnrois(str(csv_path), n_frames=T, n_rois=R)

    assert isinstance(arr, np.ndarray)
    assert arr.shape == (T, R)
    np.testing.assert_allclose(arr, df.to_numpy())


def test_read_Txnrois_ignores_index_column(tmp_path: Path):
    # Make the index non-trivial to ensure it's truly treated as index
    idx = pd.Index([100, 200, 300], name="t")
    df = pd.DataFrame({"rA": [0.1, 0.2, 0.3], "rB": [1.1, 1.2, 1.3]}, index=idx)
    p = tmp_path / "traces.csv"
    df.to_csv(p)

    arr = read_Txnrois(str(p))
    assert arr.shape == (3, 2)
    # Returned array must equal the ROI columns only (index excluded)
    np.testing.assert_allclose(arr, df.to_numpy())


def test_read_Txnrois_wrong_n_frames_raises(tmp_path: Path):
    T, R = 4, 2
    df = pd.DataFrame(
        {"roi_a": [1, 2, 3, 4], "roi_b": [5, 6, 7, 8]},
        index=pd.Index(range(T), name="time"),
    )
    p = tmp_path / "traces.csv"
    df.to_csv(p)

    with pytest.raises(ValueError) as ei:
        read_Txnrois(str(p), n_frames=T + 1, n_rois=R)
    msg = str(ei.value).lower()
    assert "expected" in msg and "frames" in msg


def test_read_Txnrois_wrong_n_rois_raises(tmp_path: Path):
    T = 6
    df = pd.DataFrame(
        {"r0": np.arange(T), "r1": np.arange(T) + 1, "r2": np.arange(T) + 2},
        index=pd.Index(range(T), name="time"),
    )
    p = tmp_path / "traces.csv"
    df.to_csv(p)

    with pytest.raises(ValueError) as ei:
        read_Txnrois(str(p), n_frames=T, n_rois=2)  # too few
    msg = str(ei.value).lower()
    assert "expected" in msg and "rois" in msg


def test_read_Txnrois_nonexistent_file_raises(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        read_Txnrois(str(missing))


def test_tiff_dask_array_matches_the_default_reader(tmp_path):
    """The page-backed array must return exactly the same data as the default reader."""
    import tifffile
    from pykrait.io.images import _tiff_dask_array

    data = np.random.default_rng(0).integers(0, 5000, (12, 2, 32, 32)).astype(np.uint16)
    path = str(tmp_path / "stack.ome.tif")
    tifffile.imwrite(path, data, metadata={"axes": "TCYX"}, compression="zstd")

    fast = _tiff_dask_array(path, expected_shape=(12, 2, 1, 32, 32))
    assert fast is not None
    assert fast.shape == (12, 2, 1, 32, 32)
    assert np.array_equal(np.asarray(fast)[:, :, 0], data)

    # and the loader returns the same values it always did
    loaded, _, _, _ = load_timelapse_lazy(path)
    assert loaded.shape == (12, 2, 32, 32)
    assert np.array_equal(np.asarray(loaded), data)


def test_tiff_dask_array_falls_back_on_shape_mismatch(tmp_path):
    import tifffile
    from pykrait.io.images import _tiff_dask_array

    data = np.random.default_rng(1).integers(0, 500, (6, 1, 16, 16)).astype(np.uint16)
    path = str(tmp_path / "stack.ome.tif")
    tifffile.imwrite(path, data, metadata={"axes": "TCYX"}, compression="zstd")

    # a shape the file does not have must not be forced onto it
    assert _tiff_dask_array(path, expected_shape=(6, 2, 1, 16, 16)) is None


def test_tiff_dask_array_returns_none_for_missing_file(tmp_path):
    from pykrait.io.images import _tiff_dask_array

    with pytest.warns(UserWarning, match="falling back to the default reader"):
        assert _tiff_dask_array(str(tmp_path / "nope.tif"), expected_shape=(1,)) is None


# --- Analysis folder resolution ---


def test_get_analysis_folder_next_to_the_video(tmp_path):
    from pykrait.io.files import get_analysis_folder

    video = tmp_path / "video.czi"
    video.write_text("stub")
    assert get_analysis_folder(str(video)) == str(tmp_path / "Analysis_video")


def test_get_analysis_folder_reuses_an_existing_analysis_folder(tmp_path):
    """Opening an intermediate result must not nest a second analysis folder."""
    from pykrait.io.files import get_analysis_folder

    analysis_folder = tmp_path / "Analysis_video"
    analysis_folder.mkdir()
    corrected = analysis_folder / "video_motion_corrected.ome.tif"
    corrected.write_text("stub")

    assert get_analysis_folder(str(corrected)) == str(analysis_folder)


def test_get_analysis_stem_drops_the_ome_suffix():
    from pykrait.io.files import get_analysis_stem

    assert get_analysis_stem("/data/video.czi") == "video"
    assert get_analysis_stem("/data/video.tif") == "video"
    assert (
        get_analysis_stem("/data/video_motion_corrected.ome.tif")
        == "video_motion_corrected"
    )


def test_create_analysis_folder_does_not_nest(tmp_path):
    """Regression: opening a motion-corrected video created a folder inside the folder."""
    from pykrait.pipeline.pipeline import AnalysisOutput, create_analysis_folder

    analysis_folder = tmp_path / "Analysis_video"
    analysis_folder.mkdir()
    corrected = analysis_folder / "video_motion_corrected.ome.tif"
    corrected.write_text("stub")

    output = create_analysis_folder(AnalysisOutput(filepath=str(corrected)))

    assert output.analysis_folder == str(analysis_folder)
    assert not (analysis_folder / "Analysis_video_motion_corrected.ome").exists()
    assert [p.name for p in analysis_folder.iterdir()] == [
        "video_motion_corrected.ome.tif"
    ]


def test_create_analysis_folder_still_creates_one_for_a_plain_video(tmp_path):
    from pykrait.pipeline.pipeline import AnalysisOutput, create_analysis_folder

    video = tmp_path / "video.czi"
    video.write_text("stub")

    output = create_analysis_folder(AnalysisOutput(filepath=str(video)))

    assert output.analysis_folder == str(tmp_path / "Analysis_video")
    assert output.filename == "video.czi"
    assert (tmp_path / "Analysis_video").is_dir()


def test_czi_reader_kwargs_follow_the_installed_bioio_czi(monkeypatch):
    """bioio-czi < 4 needs use_aicspylibczi=True, bioio-czi >= 4 rejects it."""
    import bioio_czi
    from pykrait.io.images import _czi_reader_kwargs

    class OldReader:
        def __init__(self, image, use_aicspylibczi=False, **kwargs):
            pass

    class NewReader:
        def __init__(self, image, chunk_dims="YX", **kwargs):
            pass

    monkeypatch.setattr(bioio_czi, "Reader", OldReader)
    assert _czi_reader_kwargs() == {"use_aicspylibczi": True}
    monkeypatch.setattr(bioio_czi, "Reader", NewReader)
    assert _czi_reader_kwargs() == {}


def test_imagej_metadata_is_read_unrounded_and_in_scientific_notation(tmp_path):
    import tifffile

    path = tmp_path / "fast.tif"
    tifffile.imwrite(
        path,
        np.zeros((3, 8, 8), dtype=np.uint16),
        imagej=True,
        resolution=(1 / 0.1625, 1 / 0.1625),
        metadata={"axes": "TYX", "finterval": 5e-05, "unit": "um"},
    )
    _, frame_interval, y_um, x_um = load_timelapse_lazy(str(path))
    assert frame_interval == pytest.approx(5e-05)
    assert y_um == pytest.approx(0.1625, rel=1e-4)
    assert x_um == pytest.approx(0.1625, rel=1e-4)


def test_release_tiff_handles_keeps_lazy_arrays_readable(tmp_path):
    import tifffile
    from pykrait.io.images import release_tiff_handles
    import psutil

    def open_by_this_process(path):
        target = os.path.normcase(os.path.realpath(path))
        return any(
            os.path.normcase(os.path.realpath(f.path)) == target
            for f in psutil.Process().open_files()
        )

    path = str(tmp_path / "video.tif")
    frames = np.arange(4 * 2 * 8 * 8, dtype=np.uint16).reshape(4, 2, 8, 8)
    tifffile.imwrite(path, frames, imagej=True, metadata={"axes": "TCYX"})
    lazy = load_timelapse_lazy(path)[0]
    np.testing.assert_array_equal(lazy.compute(), frames)
    assert open_by_this_process(path)

    release_tiff_handles(path)
    assert not open_by_this_process(path)
    # the array opens the file again when it is read
    np.testing.assert_array_equal(lazy.compute(), frames)
