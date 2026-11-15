import numpy as np
import pytest
from numpy.testing import assert_allclose
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

from pykrait.trace_analysis.filtering import (
    detrend_with_sinc_filter,
    _blackman_sinc_filter_kernel,
    _convolve_signals_with_kernel,
    smooth_gauss,
)


def test_blackman_sinc_filter_kernel_value_error():
    # kernel_window must be even, so passing an odd number should raise ValueError
    with pytest.raises(ValueError, match="Filter kernel window must be even"):
        _blackman_sinc_filter_kernel(kernel_window=5, cutoff_freq=0.1)


def test_convolve_signals_with_kernel_value_error():
    signals = np.random.rand(3, 100)  # dummy signals, shape (3, 100)

    # filter_kernel length must be odd, so passing even length should raise ValueError
    even_length_kernel = [0.1] * 4  # length 4 (even)
    with pytest.raises(ValueError, match="Filter kernel length must be odd"):
        _convolve_signals_with_kernel(signals, even_length_kernel)


### Testing with actual data ###
DATA_PATH = Path(__file__).parent / "detrend_test_data" / "input_signals.npy"
data = np.load(DATA_PATH)

CUTOFFS = [100, 200, 300, 400, 500]


@pytest.mark.parametrize("T_c", CUTOFFS)
def test_fft_detrending_matches_saved_pyboat(T_c):
    """
    Compares sinc_filter_pyboat_style_2d_fft with reference pyBOAT outputs
    saved earlier for different cutoff periods.
    """
    # Load test data
    input_signals = np.load(
        Path(__file__).parent / "detrend_test_data" / "input_signals.npy"
    )
    pyboat_output = np.load(
        Path(__file__).parent / "detrend_test_data" / f"pyboat_detrended_T_c_{T_c}.npy"
    )

    t = np.load(Path(__file__).parent / "detrend_test_data" / "time.npy")
    dt = t[1] - t[0]

    # Apply the sinc filter
    fft_output = detrend_with_sinc_filter(input_signals, T_c, dt)

    # Compare with precomputed pyBOAT output
    assert_allclose(
        fft_output,
        pyboat_output,
        rtol=1e-4,
        atol=1e-2,
        err_msg=f"FFT-based detrending does not match pyBOAT for T_c={T_c}",
    )


def test_returns_original_if_sigma_zero():
    data = np.random.rand(5, 100)
    output = smooth_gauss(data, 0.0)
    np.testing.assert_array_equal(output, data)


def test_returns_original_if_sigma_negative():
    data = np.random.rand(5, 100)
    output = smooth_gauss(data, -1.0)
    np.testing.assert_array_equal(output, data)


def test_output_shape_is_same_as_input():
    data = np.random.rand(10, 200)
    result = smooth_gauss(data, sigma=2.0)
    assert result.shape == data.shape


def test_smoothing_effect_is_applied():
    np.random.seed(0)
    signal = np.zeros((100, 1))
    signal[50, 0] = 1.0  # delta peak
    smoothed = smooth_gauss(signal, sigma=5.0)

    assert smoothed[50, 0] < 1.0  # peak should be lower
    assert smoothed[49, 0] > 0.0  # neighbors should have some value


def test_matches_scipy_directly():
    signal = np.random.rand(3, 50)
    sigma = 1.5
    expected = gaussian_filter1d(signal, sigma, axis=0)
    result = smooth_gauss(signal, sigma)
    np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-10)


def test_sinc_filter_interpolates_nans_per_roi():
    rng = np.random.default_rng(0)
    T = 200
    t = np.arange(T)
    signals = np.column_stack(
        [
            np.sin(2 * np.pi * t / 20) + 0.01 * t,
            np.sin(2 * np.pi * t / 30) + 0.02 * t,
            np.full(T, np.nan),
        ]
    ) + 0.01 * rng.standard_normal((T, 3))
    signals_with_nan = signals.copy()
    signals_with_nan[50, 0] = np.nan
    signals_with_nan[0, 0] = np.nan

    with pytest.warns(UserWarning, match="interpolated NaN values in 1 ROI"):
        detrended = detrend_with_sinc_filter(
            signals_with_nan, cutoff_period=100, sampling_interval=1
        )

    # the ROI with a few NaNs is fully detrended instead of becoming all-NaN
    assert np.isfinite(detrended[:, 0]).all()
    # ROIs without NaNs are unaffected
    reference = detrend_with_sinc_filter(
        signals[:, :2], cutoff_period=100, sampling_interval=1
    )
    assert_allclose(detrended[:, 1], reference[:, 1])
    # all-NaN ROIs stay NaN
    assert np.isnan(detrended[:, 2]).all()
    # the input is not modified
    assert np.isnan(signals_with_nan[50, 0])


@pytest.mark.parametrize("cutoff_period", [0, -10])
def test_sinc_filter_rejects_non_positive_cutoff(cutoff_period):
    with pytest.raises(ValueError, match="cutoff_period must be positive"):
        detrend_with_sinc_filter(np.random.rand(100, 2), cutoff_period, 1.0)


@pytest.mark.parametrize(
    "signals, message",
    [
        (np.zeros((100, 0)), "at least one ROI"),
        (np.zeros(100), "at least one ROI"),
        (np.zeros((2, 3)), "only 2 frame"),
    ],
)
def test_sinc_filter_explains_unusable_traces(signals, message):
    with pytest.raises(ValueError, match=message):
        detrend_with_sinc_filter(signals, cutoff_period=60, sampling_interval=1)
