import numpy as np
from pykrait.trace_analysis import peak_analysis

def test_find_peaks_detects_expected_peaks():
    timeseries = np.zeros((100, 1))
    timeseries[20] = 10
    timeseries[50] = 15
    timeseries[80] = 12

    result = peak_analysis.find_peaks(
        peak_min_width=1,
        peak_max_width=10,
        peak_prominence=5,
        peak_min_height=5,
        detrended_timeseries=timeseries,
    )
    assert result.shape == timeseries.shape
    detected_peaks = np.where(result[:, 0] == 1)[0]
    assert set(detected_peaks) == {20, 50, 80}

def test_calculate_normalized_peaks():
    peaks = np.zeros((100, 5))
    peaks[10, 0] = 1
    peaks[20, 1] = 1
    peaks[30, 2] = 1
    peaks[40, 3] = 1
    peaks[50, 4] = 1

    result = peak_analysis.calculate_normalized_peaks(peaks, frame_interval=1.0)
    assert result > 0
    assert isinstance(result, float)
