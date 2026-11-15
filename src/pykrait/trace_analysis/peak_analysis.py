from scipy.signal import find_peaks as scipy_find_peaks
from scipy.signal import peak_widths
import numpy as np
import pandas as pd
from typing import Tuple


def _normalize_max(trace: np.ndarray) -> np.ndarray:
    """
    scales a trace by its maximum absolute value, so that it lies between -1 and 1.
    Traces that are (almost) zero everywhere are returned unscaled.

    :param trace: 1D trace
    :type trace: np.ndarray
    :return: returns the scaled trace as float64
    :rtype: np.ndarray
    """
    trace = np.asarray(trace, dtype=np.float64)
    norm = np.max(np.abs(trace))
    if norm < 10 * np.finfo(np.float64).eps:
        norm = 1.0
    return trace / norm


def find_peaks(
    peak_min_width: int,
    peak_max_width: int,
    peak_prominence: float,
    peak_min_height: float,
    detrended_timeseries: np.ndarray,
) -> np.ndarray:
    """wrapper function for scipy's find_peaks function

    :param peak_min_width: minimum width at half-maximum of peak in frames
    :type peak_min_width: int
    :param peak_max_width: maximum width at half-maxumim of peak in frames
    :type peak_max_width: int
    :param peak_prominence: minimum prominence of peak
    :type peak_prominence: float
    :param min_peak_height: minimum height of peak
    :type min_peak_height: float
    :param detrended_timeseries: the timeseries to find the peaks on
    :type detrended_timeseries: ndarray

    :return: returns a (T x n_roi) np.ndarray with 1s at the peak locations and 0s elsewhere
    :rtype: np.ndarray
    """

    peak_series = np.zeros(detrended_timeseries.shape)

    for i in range(0, detrended_timeseries.shape[1]):
        peaks, _ = scipy_find_peaks(
            detrended_timeseries[:, i],
            width=(peak_min_width, peak_max_width),
            height=peak_min_height,
            prominence=peak_prominence,
        )
        peak_series[peaks, i] = 1

    return peak_series


def calculate_normalized_peaks(peaks: np.ndarray, frame_interval: float) -> float:
    """
    calculates the normalized peaks (peaks per 100 cells per 10 minutes)

    :param peaks: binary array of shape (T x n_cells) with 1s at the peak locations and 0s elsewhere
    :type peaks: np.ndarray
    :param frame_interval: frame interval in seconds
    :type frame_interval: float
    :return: normalized peaks per 100 cells per 10 minutes
    :rtype: float
    """
    n_frames, n_cells = peaks.shape
    n_peaks = np.sum(peaks)
    normalized_peaks = (n_peaks / (n_cells / 100)) / (
        (n_frames * frame_interval) / 600
    )  # normalize to 100 cells and 10 minutes of recording

    return normalized_peaks


def create_peak_properties(
    peak_series: np.ndarray, detrended_timeseries: np.ndarray, frame_interval: float
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    analyzes properties of the identified peaks such as FWHM, rise time, decay time, AUC and
    creates a numpy array with arranged peak traces for visualization

    The columns of the returned dataframe are:

    - cell_index: index of the ROI
    - peak_number: running number of the peak within the ROI
    - peak_loc: frame of the peak
    - peak_dist: time in seconds since the previous peak of the same ROI (NaN for the first peak)
    - FWHM: full width at half maximum in seconds
    - rise2080: time in seconds from 20% to 80% of the peak prominence on the rising flank
    - maxrise: steepest slope of the rising flank (between the left 20% point and the peak) in intensity units per second
    - decay8020: time in seconds from 80% back to 20% of the peak prominence on the decaying flank
    - maxdecay: steepest slope of the decaying flank (between the peak and the right 20% point) in intensity units per second, negative for a decay
    - AUC: area under the max-normalized trace between the 20% points (inclusive) in seconds

    peak_dist, maxrise, maxdecay and AUC are NaN where they cannot be determined.

    :param peak_series: binary array of shape (T x n_cells) with 1s at the peak locations and 0s elsewhere
    :type peak_series: np.ndarray
    :param detrended_timeseries: the detrended timeseries
    :type detrended_timeseries: np.ndarray
    :param frame_interval: frame interval in seconds
    :type frame_interval: float
    :return: returns a dataframe with peak properties and a numpy array with arranged peak traces
    :rtype: Tuple[pd.DataFrame, np.ndarray]
    """

    peak_frame_length = 70
    peak_counts = int(np.sum(peak_series, axis=None))

    peak_frame = np.zeros((peak_frame_length, peak_counts))

    # create a numpy array to hold the
    # 0 = cell index
    # 1 = peak number
    # 2 = peak_loc
    # 3 = peak_dist
    # 4 = FWHM
    # 5 = rise2080
    # 6 = maxrise
    # 7 = decay8020
    # 8 = maxdecay
    # 9 = AUC
    peak_df = np.zeros((peak_counts, 10))
    column_names = [
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

    peak_counter = 0
    # iterate over every cell
    for cell_index in range(0, peak_series.shape[1]):
        # if cell has peaks
        if np.sum(peak_series[:, cell_index]) > 0:
            # find the peak indeces
            peak_indices = np.nonzero(peak_series[:, cell_index])[0]

            # flatten the timeseries
            time_series = detrended_timeseries[:, cell_index].flatten()

            # calculate full-width half maximum
            widths, width_height, left_FWHM, right_FWHM = peak_widths(
                x=time_series, peaks=peak_indices, rel_height=0.5
            )

            # for the 20%-80% rise time and the 80%-20% decay time, calculate the indices
            # Quick Note: scipy.signal.peak_widths has weird settings for rel_height:
            # "1.0 calculates the width of the peak at its lowest contour line while 0.5
            # evaluates at half the prominence height"
            # therefore 0.8 == 20%
            # and 0.2 == 80%
            _, _, left_20, right_20 = peak_widths(
                x=time_series, peaks=peak_indices, rel_height=0.8
            )
            _, _, left_80, right_80 = peak_widths(
                x=time_series, peaks=peak_indices, rel_height=0.2
            )

            # iterate over every peak of the cell
            for peak_number in range(0, len(widths)):
                # set cell index, peak index, peak location
                peak_df[peak_counter, 0] = cell_index
                peak_df[peak_counter, 1] = peak_number
                peak_df[peak_counter, 2] = peak_indices[peak_number]
                peak_df[peak_counter, 4] = widths[peak_number] * frame_interval

                # calculate the rise time (time from 20% to 80%)
                peak_df[peak_counter, 5] = (
                    left_80[peak_number] - left_20[peak_number]
                ) * frame_interval

                # calculate the decay time (time from 80% back to 20%)
                peak_df[peak_counter, 7] = (
                    right_20[peak_number] - right_80[peak_number]
                ) * frame_interval

                # time since the previous peak of the same cell
                if peak_number > 0:
                    peak_df[peak_counter, 3] = (
                        peak_indices[peak_number] - peak_indices[peak_number - 1]
                    ) * frame_interval
                else:
                    peak_df[peak_counter, 3] = np.nan

                # calculate the AUC of the 20%-Interval using the normalized curve
                left_limit_AUC = max(int(np.floor(left_20[peak_number])), 0)
                right_limit_AUC = min(
                    int(np.ceil(right_20[peak_number])),
                    detrended_timeseries.shape[0] - 1,
                )

                # some sanity checks, the right limit is included in the interval
                if right_limit_AUC - left_limit_AUC > 0:
                    AUC = np.trapezoid(
                        y=_normalize_max(
                            time_series[left_limit_AUC : right_limit_AUC + 1]
                        ),
                        dx=frame_interval,
                    )
                    peak_df[peak_counter, 9] = AUC
                else:
                    peak_df[peak_counter, 9] = np.nan

                # steepest rise (left 20% point to peak) and steepest decay (peak to right 20% point) per second
                peak_location = peak_indices[peak_number]
                rising_flank = time_series[left_limit_AUC : peak_location + 1]
                decaying_flank = time_series[peak_location : right_limit_AUC + 1]
                peak_df[peak_counter, 6] = (
                    np.max(np.diff(rising_flank)) / frame_interval
                    if len(rising_flank) > 1
                    else np.nan
                )
                peak_df[peak_counter, 8] = (
                    np.min(np.diff(decaying_flank)) / frame_interval
                    if len(decaying_flank) > 1
                    else np.nan
                )

                # for the plot of arranged peaks:

                # set boundaries for the plot
                left_limit = int(np.floor(left_FWHM[peak_number])) - 30
                right_limit = int(np.floor(left_FWHM[peak_number])) + 40

                # if the boundaries do not clash with the start and end of the timeseries
                if left_limit > 0 and right_limit < detrended_timeseries.shape[0] - 1:
                    # normalize the trace to be between -1 and 1
                    example_trace = _normalize_max(
                        detrended_timeseries[left_limit:right_limit, cell_index]
                    )
                    peak_frame[:, peak_counter] = example_trace
                # increase peak counter to go to next row in dataframe
                peak_counter = peak_counter + 1

    # clean up, only keep frames of non-zero cells
    # Find rows where all elements are zero
    non_zero_rows = np.any(peak_frame != 0, axis=0)

    # Filter the array to remove rows with all zeros
    peak_frame_nonzero = peak_frame[:, non_zero_rows]

    # turn the numpy array of peak properties into a pandas dataframe
    peak_properties_df = pd.DataFrame(data=peak_df, columns=column_names)

    return peak_properties_df, peak_frame_nonzero
