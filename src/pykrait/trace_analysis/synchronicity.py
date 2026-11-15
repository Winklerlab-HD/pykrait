import numpy as np
from scipy import sparse
from typing import Tuple
import warnings


def find_synchronous_peaks(
    shortest_path_matrix: np.ndarray,
    peak_series: np.ndarray,
    neighbour_degree: int,
    frame_window: int,
) -> np.ndarray:
    """
    finds the number of synchronous peaks between connected cells given a certain neighbourhood degree and time window.

    :param shortest_path_matrix: ROI x ROI matrix with shortest path lengths between cells n and m
    :type shortest_path_matrix: np.ndarray
    :param peak_series: T x ROI binary matrix with 1s at calcium peak frames
    :type peak_series: np.ndarray
    :param neighbour_degree: degree of neighbourhood to consider (1 = direct neighbours)
    :type neighbour_degree: int
    :param frame_window: time window within peaks are considered synchronous
    :type frame_window: int
    :return: returns a ROI x ROI matrix with the number of synchronous peaks between connected cells
    :rtype: np.ndarray
    """

    # int64 so that cell pairs with many coincident peaks (> 255) do not overflow
    synchronous_cells_matrix = np.zeros(shortest_path_matrix.shape, dtype=np.int64)

    # find connected cells
    cell_indices = np.where(shortest_path_matrix == neighbour_degree)
    if cell_indices[0].size == 0 or cell_indices[1].size == 0:
        warnings.warn(
            f"No connected cell pairs found for the specified neighbour degree of {neighbour_degree}.",
            UserWarning,
        )
        return synchronous_cells_matrix

    peak_indices = [np.nonzero(t)[0] for t in peak_series.transpose()]

    for synchronous_cellpair in range(0, cell_indices[0].shape[0]):
        # calculates the frame-difference between all peaks of both actually neighbouring cells
        peak_differences = np.subtract.outer(
            peak_indices[cell_indices[0][synchronous_cellpair]],
            peak_indices[cell_indices[1][synchronous_cellpair]],
        )

        # some complex logical operations, built-in .__and__ basically means
        # 0 <= peak_differences <= frame_window
        synchronous_cells_matrix[
            cell_indices[0][synchronous_cellpair], cell_indices[1][synchronous_cellpair]
        ] = ((0 <= peak_differences).__and__(peak_differences <= frame_window)).sum()

        synchronous_cells_matrix[
            cell_indices[1][synchronous_cellpair], cell_indices[0][synchronous_cellpair]
        ] = ((-frame_window <= peak_differences).__and__(peak_differences < 0)).sum()

    return synchronous_cells_matrix


def find_possible_synchronous_peaks(
    shortest_path_matrix: np.ndarray,
    peak_series: np.ndarray,
    neighbour_degree: int,
    frame_window: int,
) -> np.ndarray:
    """
    finds the number of possible synchronous peaks between connected cells given a certain neighbourhood degree and time window.
    This assumes that one peak can only be synchronous with one other peak from a connected cell, so the maximum number of synchronous
    events is limited by the cell with the fewer peaks.

    :param shortest_path_matrix: ROI x ROI matrix with shortest path lengths between cells n and m
    :type shortest_path_matrix: np.ndarray
    :param peak_series: T x ROI binary matrix with 1s at calcium peak frames
    :type peak_series: np.ndarray
    :param neighbour_degree: degree of neighbourhood to consider (1 = direct neighbours)
    :type neighbour_degree: int
    :param frame_window: time window within peaks are considered synchronous
    :type frame_window: int
    :return: returns a ROI x ROI matrix with the number of possible synchronous peaks between connected cells
    :rtype: np.ndarray
    """
    possible_synchronous_cells_matrix = np.zeros(
        shortest_path_matrix.shape, dtype=np.int64
    )

    # find connected cells
    cell_indices = np.where(shortest_path_matrix == neighbour_degree)
    if cell_indices[0].size == 0 or cell_indices[1].size == 0:
        warnings.warn(
            f"No connected cell pairs found for the specified neighbour degree of {neighbour_degree}.",
            UserWarning,
        )
        return possible_synchronous_cells_matrix

    n_peaks_per_cell = np.sum(peak_series, axis=0)

    for synchronous_cellpair in range(0, cell_indices[0].shape[0]):
        possible_synchronous_cells_matrix[
            cell_indices[0][synchronous_cellpair], cell_indices[1][synchronous_cellpair]
        ] = min(
            n_peaks_per_cell[cell_indices[0][synchronous_cellpair]],
            n_peaks_per_cell[cell_indices[1][synchronous_cellpair]],
        )

    return possible_synchronous_cells_matrix


def _random_synchronous_peaks(
    shortest_path_matrix: np.ndarray,
    peak_series: np.ndarray,
    neighbour_degree: int,
    frame_window: int,
    n_iter: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    bootstraps the number of synchronous peaks between connected cells given a certain neighbourhood degree and time window by randomizing the peak series.

    :param shortest_path_matrix: ROI x ROI matrix with shortest path lengths between cells n and m
    :type shortest_path_matrix: np.ndarray
    :param peak_series: T x ROI binary matrix with 1s at calcium peak frames
    :type peak_series: np.ndarray
    :param neighbour_degree: degree of neighbourhood to consider (1 = direct neighbours)
    :type neighbour_degree: int
    :param frame_window: time window within peaks are considered synchronous
    :type frame_window: int
    :param n_iter: number of random permutations to bootstrap against, defaults to 100
    :type n_iter: int, optional
    :return: returns the number of synchronous peaks for each of the n_iter random permutations, as well as
        the ROI x ROI synchronous peaks matrix of the first permutation (as an example random control)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    count_rand_synchronous_peaks = np.zeros(shape=n_iter)
    first_rand_matrix = np.zeros(shortest_path_matrix.shape, dtype=np.int64)

    # Permuting the ROIs only reorders the rows and columns of the pairwise counts, so they
    # are counted once for all ROI pairs and looked up for every permutation. The two
    # entries find_synchronous_peaks fills for a connected cell pair together count its
    # peak pairs within +-frame_window frames, also if the pair is listed in both triangles.
    connected = shortest_path_matrix == neighbour_degree
    # a cell connected to itself is not a cell pair, it is counted the original way
    count_directly = bool(np.any(np.diagonal(connected)))
    if not count_directly:
        pairwise_counts = _count_peak_pairs_within_window(peak_series, frame_window)
        rows, columns = np.nonzero(np.triu(connected | connected.T, k=1))

    for i in range(0, n_iter):
        # randomize the peak series
        orders = np.random.permutation(np.arange(peak_series.shape[1]))

        # the matrix is only needed for the first permutation, as an example random control
        if i == 0 or count_directly:
            rand_matrix = find_synchronous_peaks(
                shortest_path_matrix=shortest_path_matrix,
                peak_series=peak_series[:, orders],
                neighbour_degree=neighbour_degree,
                frame_window=frame_window,
            )
        if i == 0:
            first_rand_matrix = rand_matrix

        if count_directly:
            count_rand_synchronous_peaks[i] = np.sum(rand_matrix, axis=None)
        else:
            count_rand_synchronous_peaks[i] = np.sum(
                pairwise_counts[orders[rows], orders[columns]]
            )

    return count_rand_synchronous_peaks, first_rand_matrix


def _count_peak_pairs_within_window(
    peak_series: np.ndarray, frame_window: int
) -> np.ndarray:
    """
    counts for every pair of ROIs the pairs of peaks that are at most frame_window frames apart.

    :param peak_series: T x ROI binary matrix with 1s at calcium peak frames
    :type peak_series: np.ndarray
    :param frame_window: time window within peaks are considered synchronous
    :type frame_window: int
    :return: returns a symmetric ROI x ROI matrix with the number of peak pairs within the window
    :rtype: np.ndarray
    """
    n_frames = peak_series.shape[0]
    peaks = sparse.csr_matrix((peak_series != 0).astype(np.int32))
    # band matrix that sums the peaks within +-frame_window frames of every frame
    offsets = range(
        -min(frame_window, n_frames - 1), min(frame_window, n_frames - 1) + 1
    )
    window = sparse.diags(
        [np.ones(n_frames - abs(offset), dtype=np.int32) for offset in offsets],
        offsets=list(offsets),
        shape=(n_frames, n_frames),
        format="csr",
    )
    return (peaks.T @ (window @ peaks)).toarray()


def calculate_synchronicity_zscore(
    shortest_path_matrix: np.ndarray,
    peak_series: np.ndarray,
    neighbour_degree: int,
    frame_window: int,
    n_iter=100,
) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    calculates the zscore of the synchronicity between connected cells given a certain neighbourhood degree and time window.

    :param shortest_path_matrix: ROI x ROI matrix with shortest path lengths between cells n and m
    :type shortest_path_matrix: np.ndarray
    :param peak_series: T x ROI binary matrix with 1s at calcium peak frames
    :type peak_series: np.ndarray
    :param neighbour_degree: degree of neighbourhood to consider (1 = direct neighbours)
    :type neighbour_degree: int
    :param frame_window: time window within peaks are considered synchronous
    :type frame_window: int
    :param n_iter: number of random permutations to bootstrap against, defaults to 100
    :type n_iter: int, optional
    :return: returns the zscore of the synchronicity, the true synchronous peaks matrix, one example random synchronous peaks matrix and the possible synchronous peaks matrix
    :rtype: Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]
    """
    # finds true connected peaks
    true_synchronous_peaks_matrix = find_synchronous_peaks(
        shortest_path_matrix, peak_series, neighbour_degree, frame_window
    )
    true_synchronous_peaks = np.sum(true_synchronous_peaks_matrix, axis=None)

    # if there are no synchronous peaks, warn the user
    if true_synchronous_peaks == 0:
        warnings.warn(
            f"No synchronous peaks found between connected cells for neighbour {neighbour_degree}. Synchronicity z-score will be NaN.",
            UserWarning,
        )
        return (
            np.nan,
            np.nan,
            np.zeros(true_synchronous_peaks_matrix.shape),
            np.zeros(true_synchronous_peaks_matrix.shape),
            np.zeros(true_synchronous_peaks_matrix.shape),
        )

    # finds random connected peaks to bootstrap against
    rand_synchronous_peaks, rand_synchronous_peaks_matrix = _random_synchronous_peaks(
        shortest_path_matrix, peak_series, neighbour_degree, frame_window, n_iter
    )

    # calculates zscore, a random control without any variance (e.g. only one possible permutation) has no defined zscore
    rand_std = np.std(rand_synchronous_peaks)
    if rand_std == 0:
        warnings.warn(
            f"The random control for neighbour {neighbour_degree} has zero variance. Synchronicity z-score will be NaN.",
            UserWarning,
        )
        true_synchronicity_zscore = np.nan
        rand_synchronicity_zscore = np.nan
    else:
        rand_mean = np.mean(rand_synchronous_peaks)
        true_synchronicity_zscore = float(
            (true_synchronous_peaks - rand_mean) / rand_std
        )
        rand_synchronicity_zscore = float(
            (rand_synchronous_peaks[0] - rand_mean) / rand_std
        )

    # calculate possible synchronous peaks matrices
    potential_synchronous_peaks_matrix = find_possible_synchronous_peaks(
        shortest_path_matrix, peak_series, neighbour_degree, frame_window
    )
    return (
        true_synchronicity_zscore,
        rand_synchronicity_zscore,
        true_synchronous_peaks_matrix,
        rand_synchronous_peaks_matrix,
        potential_synchronous_peaks_matrix,
    )
