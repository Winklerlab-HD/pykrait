import numpy as np
from pykrait.trace_analysis.peak_analysis import create_peak_properties


def _gaussian_trace(T, centers, width=3.0, height=1.0):
    t = np.arange(T)
    return sum(height * np.exp(-0.5 * ((t - c) / width) ** 2) for c in centers)


def test_create_peak_properties_with_few_cells():
    # two cells and 200 frames: the AUC check used to compare against the number of cells
    T = 200
    frame_interval = 0.5
    centers_a = [50, 120]
    centers_b = [80]
    traces = np.column_stack(
        [_gaussian_trace(T, centers_a), _gaussian_trace(T, centers_b)]
    )
    peak_series = np.zeros((T, 2))
    peak_series[centers_a, 0] = 1
    peak_series[centers_b, 1] = 1

    df, _ = create_peak_properties(peak_series, traces, frame_interval)

    assert list(df.columns) == [
        "cell_index",
        "peak_number",
        "peak_loc",
        "peak_dist",
        "FWHM",
        "rise2080",
        "maxrise",
        "decay8020",
        "maxdecay",
        "AUC",
    ]
    assert len(df) == 3
    assert (df["AUC"] > 0).all()
    assert (df["FWHM"] > 0).all()


def test_create_peak_properties_fills_dist_and_slopes():
    T = 200
    frame_interval = 0.5
    centers = [50, 120]
    trace = _gaussian_trace(T, centers)[:, None]
    peak_series = np.zeros((T, 1))
    peak_series[centers, 0] = 1

    df, _ = create_peak_properties(peak_series, trace, frame_interval)

    # time to the previous peak of the same cell, NaN for the first
    assert np.isnan(df["peak_dist"].iloc[0])
    assert np.isclose(df["peak_dist"].iloc[1], 70 * frame_interval)

    # steepest slopes per second: positive on the rise, negative on the decay
    expected_max_slope = np.max(np.diff(trace[:, 0])) / frame_interval
    assert (df["maxrise"] > 0).all()
    assert (df["maxdecay"] < 0).all()
    np.testing.assert_allclose(df["maxrise"], expected_max_slope, rtol=1e-6)
    np.testing.assert_allclose(df["maxdecay"], -expected_max_slope, rtol=1e-6)


def test_create_peak_properties_auc_matches_trapezoid():
    # a single triangular peak: the 20% points lie on the flanks, AUC uses the max-normalized trace
    T = 50
    trace = np.zeros(T)
    trace[20:31] = [0, 2, 4, 6, 8, 10, 8, 6, 4, 2, 0]
    peak_series = np.zeros((T, 1))
    peak_series[25, 0] = 1

    df, _ = create_peak_properties(peak_series, trace[:, None], frame_interval=1.0)

    # 20% of the prominence (10) is at value 2 -> frames 21 and 29, both included
    expected = np.trapezoid(trace[21:30] / 10.0, dx=1.0)
    assert np.isclose(df["AUC"].iloc[0], expected)


def test_create_peak_properties_peak_near_start():
    # the left 20% point falls on frame 0, which used to be excluded from the AUC
    T = 60
    trace = _gaussian_trace(T, [3], width=2.0)
    peak_series = np.zeros((T, 1))
    peak_series[3, 0] = 1

    df, _ = create_peak_properties(peak_series, trace[:, None], frame_interval=1.0)
    assert df["AUC"].iloc[0] > 0
