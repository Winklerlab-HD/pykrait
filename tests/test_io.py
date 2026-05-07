import pytest
import os
import datetime
import tempfile
import numpy as np
import pandas as pd
import dask.array as da
from unittest.mock import patch, MagicMock
from pykrait.io.images import load_timelapse_lazy, read_label_image, _reorder_dask_array 
from pykrait.io.files import get_files_from_folder, get_pykrait_version, save_Txnrois, read_Txnrois
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
    mock_instance.metadata.find.return_value = None # to fail regex load of metadata
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
    with pytest.warns(UserWarning, match="Failed to extract frame interval from metadata:"):
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

def test_real_tif_matches_expected():
    test_file = os.path.join("tests", "io_test_data", "iotestdata1.tif")
    assert os.path.exists(test_file), f"Test file missing: {test_file}"

    # Load actual result
    dask_arr, frame_interval, y_um, x_um = load_timelapse_lazy(test_file)
    actual_array = dask_arr.compute()

    # Load saved expected results
    expected_array = np.load("tests/io_test_data/expected_array_iotestdata1.npy")
    expected_metadata = np.load("tests/io_test_data/expected_metadata_iotestdata1.npy")

    # Compare image arrays
    assert actual_array.shape == expected_array.shape
    assert np.allclose(actual_array, expected_array, atol=1e-5), "Image array mismatch"

    # Compare metadata (frame_interval, y_um, x_um)
    actual_metadata = np.array([
        frame_interval if frame_interval is not None else -1,
        y_um if y_um is not None else -1,
        x_um if x_um is not None else -1
    ])
    assert np.allclose(actual_metadata, expected_metadata, atol=1e-2), "Metadata mismatch" # metadata is rounded to two decimal places

@patch("pykrait.io.images.BioImage")
@patch("os.path.exists", return_value=True)
def test_three_non_singleton_dims(mock_exists, mock_bioimage):
    shape = (1, 10, 1, 128, 128)
    dask_data = da.random.random(shape, chunks=(1, 1, 1, 128, 128))

    mock_instance = MagicMock()
    mock_instance.dask_data = dask_data
    mock_instance.physical_pixel_sizes = MagicMock(Y=1.0, X=1.0)
    mock_instance.metadata.find.return_value = None # to fail regex load of metadata
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
    with pytest.raises(ValueError, match=r"Failed to read label image.*RuntimeError.*Mocked failure"):
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

