import numpy as np
import warnings
import pytest
from pykrait.trace_analysis.synchronicity import (
    find_synchronous_peaks,
    _random_synchronous_peaks,
    calculate_synchronicity_zscore,
    find_possible_synchronous_peaks,
    _count_peak_pairs_within_window,
)


def test_find_synchronous_peaks_fwd():
    shortest_path_matrix = np.array([[0, 1], [1, 0]])
    peak_series = np.array(
        [
            [1, 0],  # t=0: peak in ROI 0
            [0, 1],  # t=1: peak in ROI 1
        ]
    )
    result = find_synchronous_peaks(shortest_path_matrix, peak_series, 1, 1)

    # Expect one synchronous peak between ROI0-ROI1
    expected = np.array(
        [
            [0, 0],
            [1, 0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_find_synchronous_peaks_rev():
    shortest_path_matrix = np.array([[0, 1], [1, 0]])
    peak_series = np.array(
        [
            [0, 1],  # t=0: peak in ROI 0
            [1, 0],  # t=1: peak in ROI 1
        ]
    )
    result = find_synchronous_peaks(shortest_path_matrix, peak_series, 1, 1)

    # Expect one synchronous peak between ROI0-ROI1
    expected = np.array(
        [
            [0, 1],
            [0, 0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


def test_find_synchronous_peaks_no_connections_warns():
    shortest_path_matrix = np.array(
        [
            [0, 2],
            [2, 0],
        ]
    )
    peak_series = np.eye(2)
    with warnings.catch_warnings():
        warnings.simplefilter("always")  # ensure warning is emitted
        with pytest.warns(UserWarning):
            find_synchronous_peaks(
                shortest_path_matrix, peak_series, neighbour_degree=1, frame_window=1
            )


def test_find_synchronous_peaks_large_window_fwd():
    shortest_path_matrix = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    peak_series = np.array(
        [
            [1, 0],  # t=0
            [0, 0],  # t=1
            [0, 1],  # t=2
        ]
    )
    # Window=5 should catch them even though difference=2
    result = find_synchronous_peaks(shortest_path_matrix, peak_series, 1, 5)
    print(result)
    assert result[0, 1] == 0
    assert result[1, 0] > 0


def test_find_synchronous_peaks_large_window_rev():
    shortest_path_matrix = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    peak_series = np.array(
        [
            [0, 1],  # t=0
            [0, 0],  # t=1
            [1, 0],  # t=2
        ]
    )
    # Window=5 should catch them even though difference=2
    result = find_synchronous_peaks(shortest_path_matrix, peak_series, 1, 5)
    print(result)
    assert result[0, 1] > 0
    assert result[1, 0] == 0


def test_random_synchronous_peaks_shapes():
    shortest_path_matrix = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    peak_series = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    count, matrices = _random_synchronous_peaks(
        shortest_path_matrix, peak_series, 1, 1, n_iter=10
    )

    assert count.shape == (10,)
    # only the first random matrix is returned
    assert matrices.shape == (2, 2)


def test_calculate_synchronicity_zscore_output():
    shortest_path_matrix = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    peak_series = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    z, r, true_matrix, rand_matrix, _ = calculate_synchronicity_zscore(
        shortest_path_matrix, peak_series, 1, 1, n_iter=5
    )

    # Check types and shapes
    assert isinstance(z, float)
    assert isinstance(r, float)
    assert true_matrix.shape == (2, 2)
    assert rand_matrix.shape == (2, 2)


def test_calculate_synchronicity_zscore_reproducible(monkeypatch):
    """Ensure deterministic behavior by fixing RNG."""
    shortest_path_matrix = np.array(
        [
            [0, 1, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0],
        ]
    )
    peak_series = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1],
        ]
    )

    rng = np.random.default_rng(seed=42)
    monkeypatch.setattr(np.random, "permutation", rng.permutation)

    z1, _, _, _, _ = calculate_synchronicity_zscore(
        shortest_path_matrix, peak_series, 1, 1, n_iter=1000
    )
    print(z1)

    rng = np.random.default_rng(seed=42)
    monkeypatch.setattr(np.random, "permutation", rng.permutation)
    z2, _, _, _, _ = calculate_synchronicity_zscore(
        shortest_path_matrix, peak_series, 1, 1, n_iter=1000
    )
    print(z2)

    assert np.isclose(z1, z2)


def test_calculate_synchronicity_zscore_functional():
    """Ensure deterministic behavior by fixing RNG."""
    shortest_path_matrix = np.array(
        [
            [0, 1, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 1, 0, 0],
        ]
    )
    peak_series = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1],
        ]
    )

    np.random.seed(0)
    z1, r1, _, _, _ = calculate_synchronicity_zscore(
        shortest_path_matrix, peak_series, 1, 1, n_iter=1000
    )
    print(z1)

    z2, r2, _, _, _ = calculate_synchronicity_zscore(
        shortest_path_matrix, peak_series, 1, 1, n_iter=1000
    )
    print(z1, r1)
    assert abs(z1 - z2) < 2
    assert abs(r1 - r2) < 2


def test_no_connected_cells_returns_zeros_and_warns():
    shortest_path_matrix = np.array(
        [
            [0, 2],
            [2, 0],
        ]
    )
    # 3 timepoints, 2 ROIs – values don't matter here
    peak_series = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 1],
        ]
    )

    neighbour_degree = 1  # no entries equal to 1
    frame_window = 3

    with pytest.warns(UserWarning, match="No connected cell pairs found"):
        result = find_possible_synchronous_peaks(
            shortest_path_matrix, peak_series, neighbour_degree, frame_window
        )

    # Should be all zeros and same shape as shortest_path_matrix
    assert result.shape == shortest_path_matrix.shape
    assert np.all(result == 0)
    assert result.dtype == np.int64


def test_possible_synchronous_cells_computed_as_min_peaks():
    # shortest path matrix for 3 ROIs
    # neighbours of degree 1: (0,1), (1,0), (1,2), (2,1)
    shortest_path_matrix = np.array(
        [
            [0, 1, 2],
            [1, 0, 1],
            [2, 1, 0],
        ]
    )

    # T x ROI peak series
    # peaks per ROI: [2, 2, 1]
    peak_series = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
        ]
    )

    neighbour_degree = 1
    frame_window = 3

    result = find_possible_synchronous_peaks(
        shortest_path_matrix, peak_series, neighbour_degree, frame_window
    )

    # Expected matrix:
    # (0,1) and (1,0): min(2,2) = 2
    # (1,2) and (2,1): min(2,1) = 1
    expected = np.zeros_like(shortest_path_matrix, dtype=np.int64)
    expected[0, 1] = 2
    expected[1, 0] = 2
    expected[1, 2] = 1
    expected[2, 1] = 1

    assert result.shape == shortest_path_matrix.shape
    assert result.dtype == np.int64
    np.testing.assert_array_equal(result, expected)


def test_find_synchronous_peaks_counts_above_255_do_not_overflow():
    shortest_path_matrix = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    # 300 peaks in each ROI, ROI 1 always fires one frame after ROI 0
    T = 1000
    peak_series = np.zeros((T, 2), dtype=int)
    peak_series[0:900:3, 0] = 1
    peak_series[1:901:3, 1] = 1

    result = find_synchronous_peaks(shortest_path_matrix, peak_series, 1, 1)

    assert result.dtype == np.int64
    assert result[1, 0] == 300
    assert result[0, 1] == 0


def test_calculate_synchronicity_zscore_zero_variance_returns_nan():
    # with two ROIs, every permutation gives the same number of synchronous peaks
    shortest_path_matrix = np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
    peak_series = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.warns(UserWarning, match="zero variance"):
            z, r, true_matrix, rand_matrix, _ = calculate_synchronicity_zscore(
                shortest_path_matrix, peak_series, 1, 1, n_iter=5
            )

    assert np.isnan(z)
    assert np.isnan(r)
    assert true_matrix.sum() == 1
    assert rand_matrix.shape == (2, 2)


@pytest.mark.parametrize(
    "n_frames, n_rois, frame_window", [(50, 7, 0), (50, 7, 3), (10, 5, 30), (1, 3, 2)]
)
def test_count_peak_pairs_within_window_matches_brute_force(
    n_frames, n_rois, frame_window
):
    rng = np.random.default_rng(1)
    peak_series = (rng.random((n_frames, n_rois)) < 0.2).astype(float)
    peak_indices = [np.flatnonzero(peak_series[:, roi]) for roi in range(n_rois)]
    expected = np.array(
        [
            [
                (
                    np.abs(np.subtract.outer(peak_indices[a], peak_indices[b]))
                    <= frame_window
                ).sum()
                for b in range(n_rois)
            ]
            for a in range(n_rois)
        ]
    )
    counts = _count_peak_pairs_within_window(peak_series, frame_window)
    np.testing.assert_array_equal(counts, expected)
