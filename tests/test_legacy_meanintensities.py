import numpy as np
import pytest
import warnings

from skimage.draw import disk
from pykrait.preprocessing.timeseries_extraction import _legacy_extract_mean_intensities


def generate_fake_timelapse(shape=(5, 1, 64, 64), intensity=1000):
    """Creates a fake timelapse array with constant intensity."""
    return np.full(shape, intensity)


def generate_fake_mask(shape=(64, 64), n_cells=2):
    """Creates a fake labeled mask with circular ROIs."""
    mask = np.zeros(shape, dtype=np.uint16)
    rr1, cc1 = disk((20, 20), 10)
    rr2, cc2 = disk((40, 40), 10)
    mask[rr1, cc1] = 1
    mask[rr2, cc2] = 2
    return mask


def test_extract_mean_intensities_valid_case():
    tl = generate_fake_timelapse()
    mask = generate_fake_mask()
    result = _legacy_extract_mean_intensities(tl, mask)
    assert result.shape == (5, 2)
    assert (result == 1000).all()

def test_5D_array_raises():
    tl = generate_fake_timelapse(shape=(5, 1, 3, 32, 32))
    mask = generate_fake_mask(shape=(64, 64))
    with pytest.raises(ValueError, match="Input timelapse must be a 4D array with shape TCYX"):
        _legacy_extract_mean_intensities(tl, mask)

def test_shape_mismatch_raises():
    tl = generate_fake_timelapse(shape=(5, 1, 32, 32))
    mask = generate_fake_mask(shape=(64, 64))
    with pytest.raises(ValueError, match="Masks shape.*must match the spatial dimensions"):
        _legacy_extract_mean_intensities(tl, mask)


def test_type_check_on_timelapse():
    mask = generate_fake_mask()
    with pytest.raises(TypeError):
        _legacy_extract_mean_intensities("not-an-array", mask)


def test_channel_warning():
    tl = generate_fake_timelapse(shape=(5, 2, 64, 64))  # 2 channels
    mask = generate_fake_mask()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _legacy_extract_mean_intensities(tl, mask)
        channel_warning = [warn for warn in w if "Only the first channel" in str(warn.message)]
        assert len(channel_warning) == 1


def test_unusual_shape_warning():
    tl = generate_fake_timelapse(shape=(70, 1, 64, 64))  # T > Y and T > X
    mask = generate_fake_mask(shape=(64, 64))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _legacy_extract_mean_intensities(tl, mask)
        shape_warning = [warn for warn in w if "unusual" in str(warn.message)]
        assert len(shape_warning) == 1


def test_incorrect_mask_dimensions():
    tl = generate_fake_timelapse()
    mask = np.zeros((64, 64, 2))  # 3D instead of 2D
    with pytest.raises(ValueError, match="Masks must be a 2D array"):
        _legacy_extract_mean_intensities(tl, mask)
