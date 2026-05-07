import numpy as np
import dask.array as da
import pytest
import warnings
from skimage.draw import disk
from pykrait.preprocessing.timeseries_extraction import extract_mean_intensities, _legacy_extract_mean_intensities

# --- Helpers to generate test data ---

def generate_fake_timelapse(shape=(5, 1, 64, 64), intensity=1000):
    """Creates a fake dask timelapse array with constant intensity."""
    return da.full(shape, intensity, chunks=shape)

def generate_fake_mask(shape=(64, 64), n_cells=2):
    """Creates a fake labeled mask with circular ROIs."""
    mask = np.zeros(shape, dtype=np.uint16)
    rr1, cc1 = disk((20, 20), 10)
    rr2, cc2 = disk((40, 40), 10)
    mask[rr1, cc1] = 1
    mask[rr2, cc2] = 2
    return mask


# --- Tests ---

def test_extract_mean_intensities_valid_case():
    tl = generate_fake_timelapse()
    mask = generate_fake_mask()
    result = extract_mean_intensities(tl, mask)
    assert result.shape == (5, 2)
    assert np.all(result == 1000)


def test_5D_array_raises():
    tl = generate_fake_timelapse(shape=(5, 1, 3, 32, 32))
    mask = generate_fake_mask(shape=(64, 64))
    with pytest.raises(ValueError, match="Input timelapse must be a 4D array with shape TCYX"):
        extract_mean_intensities(tl, mask)


def test_shape_mismatch_raises():
    tl = generate_fake_timelapse(shape=(5, 1, 32, 32))
    mask = generate_fake_mask(shape=(64, 64))
    with pytest.raises(ValueError, match="Masks shape.*must match the spatial dimensions"):
        extract_mean_intensities(tl, mask)


def test_type_check_on_timelapse():
    mask = generate_fake_mask()
    with pytest.raises(TypeError):
        extract_mean_intensities("not-an-array", mask)


def test_channel_warning():
    tl = generate_fake_timelapse(shape=(5, 2, 64, 64))  # 2 channels
    mask = generate_fake_mask()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        extract_mean_intensities(tl, mask)
        channel_warning = [warn for warn in w if "Input timelapse has multiple channels. Only the channel at index" in str(warn.message)]
        assert len(channel_warning) == 1


def test_unusual_shape_warning():
    tl = generate_fake_timelapse(shape=(70, 1, 64, 64))  # T > Y and T > X
    mask = generate_fake_mask(shape=(64, 64))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        extract_mean_intensities(tl, mask)
        shape_warning = [warn for warn in w if "unusual" in str(warn.message)]
        assert len(shape_warning) == 1


def test_incorrect_mask_dimensions():
    tl = generate_fake_timelapse()
    mask = np.zeros((64, 64, 2))  # 3D instead of 2D
    with pytest.raises(ValueError, match="Masks must be a 2D array"):
        extract_mean_intensities(tl, mask)

def test_legacy_and_dask_outputs_match():
    """Ensure legacy and dask-based implementations produce the same result."""
    # Generate test data
    tl_np = generate_fake_timelapse().compute()  # NumPy array for legacy
    tl_dask = da.from_array(tl_np, chunks=tl_np.shape)  # Convert back to Dask for new version
    mask = generate_fake_mask()

    # Get outputs from both functions
    legacy_result = _legacy_extract_mean_intensities(tl_np, mask)
    dask_result = extract_mean_intensities(tl_dask, mask)

    # Check for equality
    np.testing.assert_allclose(dask_result, legacy_result, err_msg="Outputs from legacy and dask versions do not match")

def test_mean_intensities_verbose():
    """Test that the verbose flag works correctly."""
    tl = generate_fake_timelapse()
    mask = generate_fake_mask()

    _ = extract_mean_intensities(tl, mask, verbose=True)
    _ = extract_mean_intensities(tl, mask, verbose=False)    


