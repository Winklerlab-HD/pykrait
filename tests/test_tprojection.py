import pytest
import numpy as np
import dask.array as da
import warnings

from pykrait.preprocessing.segmentation import (
    timelapse_projection,
    _fast_time_sum,
    _fast_time_std,
    _accumulate_projection,
    finalize_projection,
    frames_per_block,
    TemporalProjection,
)

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
    bad_array = da.from_array(
        np.zeros((2, 10, 10)), chunks=(2, 10, 10)
    )  # 3D instead of 4D
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


@pytest.mark.parametrize(
    "shape,chunks",
    [
        ((8, 2, 16, 16), (2, 2, 16, 16)),
        ((16, 1, 32, 32), (4, 1, 32, 32)),
    ],
)
def test_fast_time_reductions(shape, chunks):
    """
    Test that _fast_time_sum and _fast_time_std produce the same results as numpy.sum and numpy.std along axis=0.
    """
    # Create random NumPy data
    np_array = np.random.rand(*shape).astype(np.float32)

    # Convert to dask array
    dask_array = da.from_array(np_array, chunks=chunks)

    # Compute results using your functions, they stream the array and return numpy
    fast_sum = _fast_time_sum(dask_array)
    fast_std = _fast_time_std(dask_array)

    # Compute reference results with numpy
    expected_sum = np_array.sum(axis=0)
    expected_std = np_array.std(axis=0)

    # Assert all close within tolerance
    print(expected_std)
    print(fast_std)
    assert np.allclose(fast_sum, expected_sum, atol=1e-5), (
        "Sum result does not match numpy.sum"
    )
    assert np.allclose(fast_std, expected_std, atol=1e-5), (
        "Std result does not match numpy.std"
    )


# --- Streaming projection ---


def test_temporal_projection_matches_numpy_std():
    """The accumulator is fed block by block and must match a plain numpy reduction."""
    data = np.random.default_rng(0).normal(30000, 3, (300, 2, 16, 16)).astype(np.uint16)
    projection = TemporalProjection(method="std")
    for start in range(0, len(data), 32):
        projection.update(data[start : start + 32])
    assert projection.n_frames == len(data)
    assert np.allclose(
        projection.result(), data.astype(np.float64).std(axis=0), atol=1e-9
    )


def test_temporal_projection_matches_numpy_sum():
    data = np.random.default_rng(1).integers(0, 5000, (120, 1, 8, 8)).astype(np.uint16)
    projection = TemporalProjection(method="sum")
    for frame in data:
        projection.update(frame)  # single frames are accepted as well
    assert np.allclose(projection.result(), data.sum(axis=0, dtype=np.float64))


def test_temporal_projection_is_independent_of_block_size():
    data = np.random.default_rng(2).integers(0, 60000, (97, 2, 8, 8)).astype(np.uint16)

    results = []
    for block_size in (1, 7, 97):
        projection = TemporalProjection(method="std")
        for start in range(0, len(data), block_size):
            projection.update(data[start : start + block_size])
        results.append(projection.result())

    assert np.allclose(results[0], results[1], atol=1e-9)
    assert np.allclose(results[0], results[2], atol=1e-9)


def test_temporal_projection_is_more_accurate_than_a_naive_reduction():
    """Low variance on a high offset is where a naive sum of squares falls apart."""
    data = np.random.default_rng(3).normal(60000, 2, (500, 1, 16, 16)).astype(np.uint16)
    exact = data.astype(np.float64).std(axis=0)

    projection = TemporalProjection(method="std")
    projection.update(data)

    assert np.abs(projection.result() - exact).max() < 1e-6 * exact.mean()


def test_temporal_projection_rejects_unknown_method():
    with pytest.raises(ValueError):
        TemporalProjection(method="median")


def test_temporal_projection_without_frames():
    with pytest.raises(ValueError):
        TemporalProjection(method="std").result()


def test_frames_per_block_fits_the_budget():
    timelapse = da.zeros((1000, 2, 512, 512), dtype=np.uint16)
    frame_bytes = 2 * 512 * 512 * 2
    assert frames_per_block(timelapse, memory_budget=10 * frame_bytes) == 10
    # never returns less than one frame, however small the budget
    assert frames_per_block(timelapse, memory_budget=1) == 1


def test_finalize_projection_scales_and_normalizes():
    raw = np.random.default_rng(4).random((2, 32, 32)) * 100
    plain = finalize_projection(raw, normalize=False)
    enhanced = finalize_projection(raw, normalize=True)
    assert plain.dtype == np.uint16 and enhanced.dtype == np.uint16
    assert plain.shape == enhanced.shape == raw.shape
    assert plain.max() == 65535  # rescaled to the full range


def test_projection_streams_without_holding_the_timelapse(monkeypatch):
    """Blocks must be requested one at a time, never the whole timelapse at once."""
    data = np.random.default_rng(5).integers(0, 5000, (64, 1, 8, 8)).astype(np.uint16)
    timelapse = da.from_array(data, chunks=(1, 1, 8, 8))

    requested = []
    original_asarray = np.asarray

    def tracking_asarray(obj, *args, **kwargs):
        if hasattr(obj, "shape") and getattr(obj, "ndim", 0) == 4:
            requested.append(obj.shape[0])
        return original_asarray(obj, *args, **kwargs)

    monkeypatch.setattr(
        "pykrait.preprocessing.segmentation.np.asarray", tracking_asarray
    )
    monkeypatch.setattr(
        "pykrait.preprocessing.segmentation.frames_per_block", lambda *a, **k: 16
    )
    _accumulate_projection(timelapse, method="std")

    # every read covers at most one block, the full 64 frames are never materialized
    assert requested
    assert max(requested) == 16


# --- Sum projection: overflow and contrast enhancement ---


def test_sum_projection_does_not_overflow_16_bit():
    """The per-pixel sum runs far past the range of the input dtype."""
    data = np.full((200, 1, 8, 8), 60000, dtype=np.uint16)
    result = _fast_time_sum(da.from_array(data, chunks=(1, 1, 8, 8)))

    assert result[0, 0, 0] == 200 * 60000  # 12_000_000, uint16 would wrap at 65_535
    assert np.array_equal(result, data.astype(np.float64).sum(axis=0))


def test_sum_projection_does_not_overflow_32_bit():
    """Long recordings push the sum past 32-bit as well."""
    n_frames = 70_000  # 70_000 * 65_535 > 2**32
    data = np.full((n_frames, 1, 2, 2), 65535, dtype=np.uint16)
    projection = TemporalProjection(method="sum")
    for start in range(0, n_frames, 10_000):
        projection.update(data[start : start + 10_000])

    expected = n_frames * 65535
    assert projection.result()[0, 0, 0] == expected
    assert expected > 2**32  # the accumulator has to hold more than 32 bits


def test_sum_projection_is_exact_for_integer_input():
    """float64 represents integer sums exactly well past any realistic recording."""
    data = np.random.default_rng(0).integers(0, 65535, (500, 2, 8, 8)).astype(np.uint16)
    result = _fast_time_sum(da.from_array(data, chunks=(1, 1, 8, 8)))
    assert np.array_equal(result, data.astype(np.float64).sum(axis=0))


def test_sum_projection_with_clahe():
    """Regression: the raw sum is far outside the range CLAHE accepts."""
    data = (
        np.random.default_rng(1).integers(0, 60000, (50, 2, 32, 32)).astype(np.uint16)
    )
    lazy = da.from_array(data, chunks=(1, 1, 32, 32))

    result = timelapse_projection(lazy, method="sum", normalize=True, verbose=False)

    assert result.dtype == np.uint16
    assert result.shape == (2, 32, 32)


def test_sum_and_std_projections_both_return_16_bit(dummy_timelapse):
    for method in ("std", "sum"):
        for normalize in (False, True):
            result = timelapse_projection(
                dummy_timelapse, method=method, normalize=normalize, verbose=False
            )
            assert result.dtype == np.uint16, (method, normalize)
            assert result.shape == (2, 10, 10)
