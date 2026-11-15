import numpy as np
import pytest
from pykrait.trace_analysis.oscillations import (
    calculate_std_cov,
    get_random_std_covs,
    find_oscillating_rois,
)
from pykrait.trace_analysis import oscillations


def _col_from_peaks(T, peak_indices):
    """Build one ROI column (length T) with 1s at given indices."""
    col = np.zeros(T, dtype=int)
    col[peak_indices] = 1
    return col


def test_returns_nan_when_fewer_than_4_peaks():
    """
    With the current implementation:
      len(diffs) < 4  -> NaN
      diffs = np.diff(peak_indices), so we need at least 5 peaks to compute.
    """
    T = 100
    # ROIs: 0 peaks, 1 peak, 2 peaks, 3 peaks, 4 peaks (all should be NaN)
    cols = [
        _col_from_peaks(T, []),
        _col_from_peaks(T, [10]),
        _col_from_peaks(T, [10, 30]),
        _col_from_peaks(T, [10, 30, 60]),
    ]
    peak_series = np.stack(cols, axis=1)
    sd, cov = calculate_std_cov(peak_series, frame_interval=1.0)

    assert sd.shape == (4,)
    assert cov.shape == (4,)
    assert np.all(np.isnan(sd))
    assert np.all(np.isnan(cov))


def test_equal_intervals_with_5_peaks_gives_zero_sd_and_cov():
    """
    5 peaks -> 4 intervals; equal spacing => std=0, cov=0.
    """
    T = 200
    peaks = [0, 10, 20, 30, 40]  # 4 equal intervals of 10 frames
    peak_series = _col_from_peaks(T, peaks)[:, None]  # (T, 1)
    sd, cov = calculate_std_cov(peak_series, frame_interval=1.0)

    assert sd.shape == (1,)
    assert cov.shape == (1,)
    assert np.isclose(sd[0], 0.0)
    assert np.isclose(cov[0], 0.0)


def test_unequal_intervals_numeric_check():
    """
    5 peaks -> 4 intervals; check numeric std and cov against NumPy.
    """
    T = 300
    peaks = [0, 10, 25, 45, 70]  # intervals (frames): [10, 15, 20, 25]
    frame_interval = 1.0  # seconds per frame
    diffs_sec = np.diff(peaks) * frame_interval
    expected_sd = np.std(diffs_sec)
    expected_cov = expected_sd / np.mean(diffs_sec)

    peak_series = _col_from_peaks(T, peaks)[:, None]
    sd, cov = calculate_std_cov(peak_series, frame_interval=frame_interval)

    assert np.isclose(sd[0], expected_sd)
    assert np.isclose(cov[0], expected_cov)


def test_frame_interval_scales_intervals():
    """
    Changing frame_interval should scale the std (and mean) proportionally.
    """
    T = 300
    peaks = [0, 10, 25, 45, 70]  # intervals (frames): [10, 15, 20, 25]
    fi_a = 1.0
    fi_b = 0.5  # half the seconds per frame -> all intervals halved in seconds

    peak_series = _col_from_peaks(T, peaks)[:, None]
    sd_a, cov_a = calculate_std_cov(peak_series, frame_interval=fi_a)
    sd_b, cov_b = calculate_std_cov(peak_series, frame_interval=fi_b)

    # std scales with frame_interval
    assert np.isclose(sd_b[0], sd_a[0] * (fi_b / fi_a))
    # cov = std / mean -> scale cancels out -> invariant
    assert np.isclose(cov_b[0], cov_a[0])


def test_random_std_covs_shapes_and_finite():
    """
    For ROIs with ≥5 peaks, the randomized control should:
    - return arrays of length n_roi * n_iter
    - be finite (no NaNs), since each column has ≥4 intervals after shuffling
    - generally not be all zeros (since shuffling breaks equal spacing)
    """
    T = 100
    # Two ROIs, each with 5 peaks (=> 4 intervals)
    peaks_a = [0, 20, 40, 60, 80]
    peaks_b = [10, 35, 55, 75, 95]
    peak_series = np.stack(
        [
            _col_from_peaks(T, peaks_a),
            _col_from_peaks(T, peaks_b),
        ],
        axis=1,  # (T, n_roi)
    )

    n_iter = 3
    frame_interval = 1.0

    sd, cov = get_random_std_covs(
        peak_series, frame_interval=frame_interval, n_iter=n_iter
    )

    assert sd.shape == (peak_series.shape[1] * n_iter,)
    assert cov.shape == (peak_series.shape[1] * n_iter,)

    # Should be finite given enough intervals
    assert np.isfinite(sd).all()
    assert np.isfinite(cov).all()

    # Not all zeros (very unlikely after shuffling equal spacings)
    assert not np.allclose(sd, 0.0)


def test_random_std_covs_deterministic_with_seed():
    """
    With a fixed-seed Generator, the shuffle can be reproduced externally:
    each iteration shuffles every column independently along the time axis.
    """
    T = 50
    # Single ROI with equal spacing: 5 peaks -> 4 intervals
    peaks = [0, 10, 20, 30, 40]
    filtered_peak_series = _col_from_peaks(T, peaks)[:, None]  # (T, 1)

    n_iter = 4
    frame_interval = 1.25

    sd, cov = get_random_std_covs(
        filtered_peak_series,
        frame_interval=frame_interval,
        n_iter=n_iter,
        rng=np.random.default_rng(123),
    )

    # Build expected result using the SAME shuffle recipe with a separate RNG of the same seed
    rng_for_expected = np.random.default_rng(123)
    shuffled = [
        rng_for_expected.permuted(filtered_peak_series != 0, axis=0)
        for _ in range(n_iter)
    ]
    expected_sd, expected_cov = calculate_std_cov(
        np.hstack(shuffled), frame_interval=frame_interval
    )

    np.testing.assert_allclose(sd, expected_sd, rtol=0, atol=0)
    np.testing.assert_allclose(cov, expected_cov, rtol=0, atol=0)

    # a second call with the same seed gives the same result
    sd2, cov2 = get_random_std_covs(
        filtered_peak_series,
        frame_interval=frame_interval,
        n_iter=n_iter,
        rng=np.random.default_rng(123),
    )
    np.testing.assert_array_equal(sd, sd2)
    np.testing.assert_array_equal(cov, cov2)


def test_random_std_covs_preserves_peak_counts():
    T = 100
    peak_series = np.stack(
        [_col_from_peaks(T, [0, 20, 40, 60, 80]), _col_from_peaks(T, [1, 2, 3])], axis=1
    )
    sd, cov = get_random_std_covs(
        peak_series, frame_interval=1.0, n_iter=5, rng=np.random.default_rng(0)
    )
    # ordered by iteration, then ROI: the ROI with 3 peaks (2 intervals) is always NaN
    assert sd.shape == (10,)
    assert np.isfinite(sd[0::2]).all()
    assert np.isnan(sd[1::2]).all()


def _dummy_peak_series(T=100, n_roi=3):
    """
    Build a small (T, n_roi) peak_series where each ROI has >=5 peaks (=> 4 intervals).
    """
    peaks_list = [
        [0, 20, 40, 60, 80],  # equal spacing
        [5, 30, 50, 70, 95],  # slightly offset
        [10, 25, 45, 65, 90],  # uneven
    ][:n_roi]
    cols = [_col_from_peaks(T, p) for p in peaks_list]
    return np.stack(cols, axis=1)


def test_invalid_method_raises():
    peak_series = _dummy_peak_series()
    with pytest.raises(ValueError):
        find_oscillating_rois(
            periodicity_method="nope",
            peak_series=peak_series,
            std_threshold=1.0,
            cov_threshold=0.5,
            frame_interval=1.0,
        )


def test_invalid_cov_threshold_raises():
    peak_series = _dummy_peak_series()
    # cov_threshold must be in [0, 1] when it is a quantile
    with pytest.raises(ValueError):
        find_oscillating_rois(
            periodicity_method="quantile",
            peak_series=peak_series,
            std_threshold=0.5,
            cov_threshold=1.1,
            frame_interval=1.0,
        )


def test_cov_cutoff_above_one_is_allowed():
    peak_series = _dummy_peak_series()
    # as a cutoff, a CoV above 1 is a valid threshold
    results, _, _ = find_oscillating_rois(
        periodicity_method="cutoff",
        peak_series=peak_series,
        std_threshold=1.0,
        cov_threshold=1.1,
        frame_interval=1.0,
        n_iter=5,
    )
    assert results["cov_cutoff"] == 1.1
    assert results["experimental_below_cov"] == 3


def test_invalid_std_quantile_raises_in_quantile_branch():
    peak_series = _dummy_peak_series()
    # std_threshold in "quantile" branch must be in [0, 1]
    with pytest.raises(ValueError):
        find_oscillating_rois(
            periodicity_method="quantile",
            peak_series=peak_series,
            std_threshold=1.5,
            cov_threshold=0.5,
            frame_interval=1.0,
        )


def _col_from_peaks(T, idx):
    col = np.zeros(T, dtype=int)
    col[idx] = 1
    return col


def test_cutoff_branch_handcrafted(monkeypatch):
    """
    Hand-crafted example (no randomness in expectations).
    Two ROIs, both with 5 peaks => 4 intervals each.
      ROI1 peaks: [0,10,20,30,40] -> diffs = [10,10,10,10] -> std=0, cov=0
      ROI2 peaks: [0,10,30,60,100] -> diffs = [10,20,30,40]
        mean = 25, var = 125, std = sqrt(125) = 5*sqrt(5) ≈ 11.180339887
        cov  = std/mean = (5*sqrt(5))/25 = sqrt(5)/5 ≈ 0.4472135955
    We choose thresholds so counts are obvious:
      std_threshold = 5.0  -> ROI1 below, ROI2 above => experimental_below_std = 1
      cov_threshold = 0.2  -> ROI1 below, ROI2 above => experimental_below_cov = 1

    For the random-control quantiles, we return simple arrays so we can compute by hand:
      random_stds_for_quantile = [0, 1, 2, 10, 20]   -> <=5 are 3/5 => std_quantile = 0.6
      random_covs_for_quantile = [0, 0.1, 0.2, 0.4, 0.8] -> <=0.2 are 3/5 => cov_quantile = 0.6

    For the random_below_* counts (the second random call with n_iter=1):
      random_stds = [0, 6]     -> <=5 -> 1
      random_covs = [0.2, 0.3] -> <=0.2 -> 1
    """
    T = 120
    roi1 = _col_from_peaks(T, [0, 10, 20, 30, 40])
    roi2 = _col_from_peaks(T, [0, 10, 30, 60, 100])
    peak_series = np.stack([roi1, roi2], axis=1)  # (T, 2)

    frame_interval = 1.0
    std_threshold = 5.0
    cov_threshold = 0.2
    n_iter = 7  # any value; we assert behavior, not the size

    # Monkeypatch only the random-control function to deterministic arrays
    calls = []

    def fake_get_random_std_covs(ps, fi, n_iter=100):
        calls.append(n_iter)
        if len(calls) == 1:
            return np.array([0.0, 1.0, 2.0, 10.0, 20.0]), np.array(
                [0.0, 0.1, 0.2, 0.4, 0.8]
            )
        else:
            return np.array([0.0, 6.0]), np.array([0.2, 0.3])

    monkeypatch.setattr(oscillations, "get_random_std_covs", fake_get_random_std_covs)
    # Run function
    results, experimental_stds, experimental_covs = find_oscillating_rois(
        periodicity_method="cutoff",
        peak_series=peak_series,
        std_threshold=std_threshold,
        cov_threshold=cov_threshold,
        frame_interval=frame_interval,
        n_iter=n_iter,
    )

    assert calls == [n_iter, 1]
    # Hand-computed experimental stats
    std_roi1, cov_roi1 = 0.0, 0.0
    std_roi2 = np.sqrt(125.0)  # ≈ 11.180339887
    cov_roi2 = std_roi2 / 25.0  # ≈ 0.4472135955

    np.testing.assert_allclose(experimental_stds, [std_roi1, std_roi2])
    np.testing.assert_allclose(experimental_covs, [cov_roi1, cov_roi2])

    # Pass-through cutoffs
    assert results["std_cutoff"] == std_threshold
    assert results["cov_cutoff"] == cov_threshold

    # Quantiles from hand arrays: <=5 among [0,1,2,10,20] = 3/5; <=0.2 among [0,0.1,0.2,0.4,0.8] = 3/5
    print("std_quantile", results["std_quantile"])
    assert np.isclose(results["std_quantile"], 0.6)
    assert np.isclose(results["cov_quantile"], 0.6)

    # Counts (<= with NaN ignored; none here)
    # experimental
    assert results["experimental_below_std"] == 1  # ROI1 only
    assert results["experimental_below_cov"] == 1  # ROI1 only
    # random (from second call)
    assert results["random_below_std"] == 1  # 0.0 <= 5.0 (6.0 > 5.0)
    assert results["random_below_cov"] == 1  # 0.2 <= 0.2 (0.3 > 0.2)


def test_quantile_branch_handcrafted(monkeypatch):
    """
    Quantile branch: we hand-pick the random-control arrays so np.nanquantile is trivial to compute.

    Use the SAME peak_series as above, so experimental stats are:
      stds: [0, sqrt(125)], covs: [0, sqrt(125)/25]

    For quantile inputs:
      std_quantile = 0.6, cov_quantile = 0.6

    Random-control arrays for cutoffs (first call):
      random_stds_for_q = [0, 1, 2, 10, 20]
        0.6-quantile (NumPy default): index = 4*0.6 = 2.4 between v2=2 and v3=10:
          std_cutoff = 2 + 0.4*(10-2) = 5.2
      random_covs_for_q = [0.0, 0.1, 0.2, 0.4, 0.8]
        cov_cutoff = 0.2 + 0.4*(0.4-0.2) = 0.28

    Second call arrays for counts:
      random_stds = [0, 6]   -> <= 5.2 => 1
      random_covs = [0.2, 0.3] -> <= 0.28 => 1

    Experimental counts vs cutoffs:
      stds: [0, 11.1803] <= 5.2 => 1
      covs: [0, 0.4472]  <= 0.28 => 1
    """
    T = 120
    roi1 = _col_from_peaks(T, [0, 10, 20, 30, 40])
    roi2 = _col_from_peaks(T, [0, 10, 30, 60, 100])
    peak_series = np.stack([roi1, roi2], axis=1)

    frame_interval = 1.0
    std_q = 0.6
    cov_q = 0.6
    n_iter = 13

    calls = []

    def fake_get_random_std_covs(ps, fi, n_iter=100):
        calls.append(n_iter)
        if len(calls) == 1:
            return np.array([0.0, 1.0, 2.0, 10.0, 20.0]), np.array(
                [0.0, 0.1, 0.2, 0.4, 0.8]
            )
        else:
            return np.array([0.0, 6.0]), np.array([0.2, 0.3])

    monkeypatch.setattr(oscillations, "get_random_std_covs", fake_get_random_std_covs)

    results, experimental_stds, experimental_covs = find_oscillating_rois(
        periodicity_method="quantile",
        peak_series=peak_series,
        std_threshold=std_q,
        cov_threshold=cov_q,
        frame_interval=frame_interval,
        n_iter=n_iter,
    )

    # n_iter then 1
    assert calls == [n_iter, 1]

    # Hand experimental values
    std_roi2 = np.sqrt(125.0)
    cov_roi2 = std_roi2 / 25.0

    np.testing.assert_allclose(experimental_stds, [0.0, std_roi2])
    np.testing.assert_allclose(experimental_covs, [0.0, cov_roi2])

    # Cutoffs by hand:
    assert np.isclose(results["std_cutoff"], 5.2)
    assert np.isclose(results["cov_cutoff"], 0.28)

    # Quantiles echoed back
    assert np.isclose(results["std_quantile"], std_q)
    assert np.isclose(results["cov_quantile"], cov_q)

    # Counts (<=; NaNs ignored)
    assert results["experimental_below_std"] == 1
    assert results["experimental_below_cov"] == 1
    assert results["random_below_std"] == 1
    assert results["random_below_cov"] == 1


def test_find_median_frequency_values_and_types():
    from pykrait.trace_analysis.oscillations import find_median_frequency

    T = 100
    peak_series = np.stack(
        [
            _col_from_peaks(T, [0, 10, 20, 30]),  # 3 intervals of 10 frames
            _col_from_peaks(T, [0, 20, 40, 60]),  # 3 intervals of 20 frames
            _col_from_peaks(T, [5, 50]),  # 1 interval, not an active cell
        ],
        axis=1,
    )
    total_mHz, active_mHz, periods = find_median_frequency(
        peak_series, frame_interval=2.0
    )

    # all intervals in seconds: [20, 20, 20, 40, 40, 40, 90] -> median 40 s -> 25 mHz
    assert np.isclose(total_mHz, 25.0)
    # cell periods: [20, 40, NaN] -> median 30 s
    assert isinstance(periods, np.ndarray)
    np.testing.assert_allclose(periods, [20.0, 40.0, np.nan])
    assert np.isclose(active_mHz, 1000 / 30)


def test_find_median_frequency_without_active_cells_is_nan():
    from pykrait.trace_analysis.oscillations import find_median_frequency
    import warnings

    peak_series = np.zeros((50, 2), dtype=int)
    peak_series[[5, 25], 0] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        total_mHz, active_mHz, periods = find_median_frequency(
            peak_series, frame_interval=1.0
        )
    assert np.isclose(total_mHz, 50.0)
    assert np.isnan(active_mHz)
    assert np.isnan(periods).all()
