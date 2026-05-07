import pytest
import numpy as np
import dask.array as da
import warnings

from pykrait.preprocessing.segmentation import timelapse_projection, _fast_time_sum, _fast_time_std

# --- Fixtures / helpers ---

@pytest.fixture
def dummy_timelapse():
    """Returns a simple 4D dask array with shape (T, C, Y, X) = (5, 2, 10, 10)."""
    np_data = np.random.rand(5, 2, 10, 10).astype(np.float32)
    return da.from_array(np_data, chunks=(1, 2, 10, 10))

@pytest.fixture
def constant_timelapse():
    """Returns a 4D array where all values are constant (edge case for std=0)."""
    return da.from_array(np.ones((4, 1, 8, 8)), chunks=(2, 1, 8, 8))

# --- Tests ---

def test_std_projection_output_shape_and_type(dummy_timelapse):
    result = timelapse_projection(dummy_timelapse, method="std", normalize=False)
    assert result.shape == (2, 10, 10)
    assert result.dtype == np.uint16

def test_sum_projection_output(dummy_timelapse):
    result = timelapse_projection(dummy_timelapse, method="sum", normalize=False)
    assert result.shape == (2, 10, 10)
    assert result.dtype == np.uint16

def test_normalization_flag(dummy_timelapse):
    result = timelapse_projection(dummy_timelapse, normalize=True, method="std")
    assert result.shape == (2, 10, 10)
    assert result.dtype == np.uint16
    # Check values are contrast-enhanced
    assert result.max() <= 65535 and result.min() >= 0

def test_constant_data_std_projection(constant_timelapse):
    result = timelapse_projection(constant_timelapse, method="std", normalize=False)
    # std across time = 0 → should return all zeros
    assert np.all(result == 0)

def test_invalid_input_type():
    with pytest.raises(TypeError):
        timelapse_projection(np.zeros((5, 2, 10, 10)), method="sum")

def test_invalid_shape():
    bad_array = da.from_array(np.zeros((2, 10, 10)), chunks=(2, 10, 10))  # 3D instead of 4D
    with pytest.raises(ValueError):
        timelapse_projection(bad_array, method="sum")

def test_invalid_method(dummy_timelapse):
    with pytest.raises(ValueError):
        timelapse_projection(dummy_timelapse, method="median")

def test_warns_on_unusual_shape_too_large():
    # T is much larger than Y or X — unusual
    bad_shape = da.from_array(np.random.rand(100, 1, 8, 8), chunks=(10, 1, 8, 8))
    with pytest.warns(UserWarning, match="unusual"):
        timelapse_projection(bad_shape, method="sum", normalize=False)

def test_warns_on_unusual_shape_c_greater_than_yx():
    # C is larger than Y and X — unusual
    bad_shape = da.from_array(np.random.rand(4, 64, 8, 8), chunks=(2, 64, 8, 8))
    with pytest.warns(UserWarning, match="unusual"):
        timelapse_projection(bad_shape, method="std", normalize=False)

def test_no_warning_on_normal_shape(dummy_timelapse):
    # Should not trigger a warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        timelapse_projection(dummy_timelapse, method="std", normalize=False)
        assert not any(isinstance(item.category, UserWarning) for item in w)


@pytest.mark.parametrize("shape,chunks", [
    ((8, 2, 16, 16), (2, 2, 16, 16)),
    ((16, 1, 32, 32), (4, 1, 32, 32)),
])
def test_fast_time_reductions(shape, chunks):
    """
    Test that _fast_time_sum and _fast_time_std produce the same results as numpy.sum and numpy.std along axis=0.
    """
    # Create random NumPy data
    np_array = np.random.rand(*shape).astype(np.float32)

    # Convert to dask array
    dask_array = da.from_array(np_array, chunks=chunks)

    # Compute results using your functions
    fast_sum = _fast_time_sum(dask_array).compute() 
    fast_std = _fast_time_std(dask_array).compute()

    # Compute reference results with numpy
    expected_sum = np_array.sum(axis=0)
    expected_std = np_array.std(axis=0)

    # Assert all close within tolerance
    print(expected_std)
    print(fast_std)
    assert np.allclose(fast_sum, expected_sum, atol=1e-5), "Sum result does not match numpy.sum"
    assert np.allclose(fast_std, expected_std, atol=1e-5), "Std result does not match numpy.std"
