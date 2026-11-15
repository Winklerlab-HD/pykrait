"""
Regression test of the analysis results.

Runs every analysis step after the intensity extraction on a real recording (548 ROIs,
400 frames) with a fixed seed and compares the results with the ones saved in
``pipeline_test_data/expected_results.npz`` and ``expected_output.json``. Refactorings
must reproduce them, up to rounding differences in the last bits of floats.

If a change is meant to change the results, regenerate the expected results with
``python tests/test_regression.py`` and explain the change in the commit message.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pykrait.io.files import read_Txnrois
from pykrait.io.images import read_label_image
from pykrait.pipeline.pipeline import (
    AnalysisOutput,
    AnalysisParameters,
    calculate_periodic_cells,
    calculate_synchronicity,
    create_roi_dataframe,
    do_peak_analysis,
    save_analysis_results,
)
from pykrait.preprocessing.timeseries_extraction import (
    extract_cell_properties,
    get_adjacency_matrix,
)
from pykrait.trace_analysis.filtering import detrend_with_sinc_filter
from pykrait.trace_analysis.synchronicity import calculate_synchronicity_zscore

DATA = Path(__file__).parent / "pipeline_test_data"
EXPECTED_ARRAYS = DATA / "expected_results.npz"
EXPECTED_OUTPUT = DATA / "expected_output.json"
SEED = 0
# FFTs and BLAS differ in the last bits between platforms and numpy/scipy versions
# (relative differences of about 1e-10 were seen), counts are compared exactly
RTOL = 1e-9

# the parameters the recording was analysed with, see analysis_parameters.csv
PARAMETERS = AnalysisParameters(
    neighbour_tolerance=5,
    peak_max_width=80,
    peak_min_height=800,
    peak_prominence=1000,
)
FRAME_INTERVAL = 1.260749
PIXEL_INTERVAL = 0.69

# fields that describe the run rather than its results
UNCOMPARED_FIELDS = {"pykrait_version", "timestamp", "filepath", "analysis_folder"}


def run_analysis():
    """Runs the analysis steps of CalciumVideo.run that follow the intensity extraction."""
    raw_intensities = np.load(DATA / "numpy_arrays.npz")["raw_intensities"]
    masks = read_label_image(str(DATA / "cp_masks.ome.tif"))
    params = AnalysisParameters(**vars(PARAMETERS))
    output = AnalysisOutput(
        filename="recording.czi",
        frame_interval=FRAME_INTERVAL,
        pixel_interval_x=PIXEL_INTERVAL,
        pixel_interval_y=PIXEL_INTERVAL,
    )
    output.number_of_frames, output.number_of_cells = raw_intensities.shape

    np.random.seed(SEED)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        detrended = detrend_with_sinc_filter(
            raw_intensities, params.sinc_filter_window, FRAME_INTERVAL
        )
        peaks, peak_properties, peak_traces, output = do_peak_analysis(
            detrended,
            int(params.peak_min_width / FRAME_INTERVAL),
            int(params.peak_max_width / FRAME_INTERVAL),
            params,
            output,
        )
        output, stds, covs = calculate_periodic_cells(peaks, params, output)
        roi_dataframe = create_roi_dataframe(peaks, stds, covs, output)
        cell_properties, cell_positions = extract_cell_properties(masks)
        adjacency, shortest_paths = get_adjacency_matrix(
            masks, int(params.neighbour_tolerance / PIXEL_INTERVAL), verbose=False
        )
        output, true_synchronous, random_synchronous = calculate_synchronicity(
            shortest_paths, peaks, params, output
        )
        # the z-scores in the output are rounded, so they are kept unrounded as well
        np.random.seed(SEED)
        frame_window = int(params.time_window / FRAME_INTERVAL)
        zscores = np.array(
            [
                calculate_synchronicity_zscore(
                    shortest_paths, peaks, degree, frame_window * degree
                )[:2]
                for degree in (1, 2, 3)
            ]
        )

    arrays = {
        "raw_intensities": raw_intensities,
        "masks": masks,
        "detrended": detrended,
        "peaks": peaks,
        "peak_properties": peak_properties.to_numpy(),
        "peak_traces": peak_traces,
        "stds": stds,
        "covs": covs,
        "roi_dataframe": roi_dataframe.to_numpy(dtype=float),
        "cell_properties": cell_properties.to_numpy(),
        "cell_positions": cell_positions,
        "adjacency": adjacency,
        "shortest_paths": shortest_paths,
        "true_synchronous": true_synchronous,
        "random_synchronous": random_synchronous,
        "zscores": zscores,
    }
    return params, output, arrays, roi_dataframe


def _output_dict(output: AnalysisOutput) -> dict:
    """Converts the comparable fields of an AnalysisOutput into plain Python values."""
    result = {}
    for key, value in vars(output).items():
        if key in UNCOMPARED_FIELDS:
            continue
        if isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


@pytest.fixture(scope="module")
def analysis():
    return run_analysis()


def test_analysis_output_is_unchanged(analysis):
    _, output, _, _ = analysis
    expected = json.loads(EXPECTED_OUTPUT.read_text())
    actual = _output_dict(output)
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if isinstance(value, float):
            assert actual[key] == pytest.approx(value, rel=RTOL, nan_ok=True), key
        else:
            assert actual[key] == value, key


@pytest.mark.parametrize(
    "name",
    [
        "detrended",
        "peaks",
        "peak_properties",
        "peak_traces",
        "stds",
        "covs",
        "roi_dataframe",
        "cell_properties",
        "cell_positions",
        "adjacency",
        "shortest_paths",
        "true_synchronous",
        "random_synchronous",
        "zscores",
    ],
)
def test_analysis_arrays_are_unchanged(analysis, name):
    _, _, arrays, _ = analysis
    expected = np.load(EXPECTED_ARRAYS)[name]
    assert arrays[name].shape == expected.shape
    np.testing.assert_allclose(arrays[name], expected, rtol=RTOL, atol=0)


def test_saved_results_match_the_analysis(analysis, tmp_path: Path):
    params, output, arrays, roi_dataframe = analysis
    output.filepath = str(tmp_path / "recording.czi")
    save_analysis_results(
        tproj=arrays["masks"].astype(np.uint16),
        masks=arrays["masks"],
        synchronous_peaks_matrix=arrays["true_synchronous"],
        analysis_output=output,
        analysis_params=params,
        detrended_intensities=arrays["detrended"],
        peaks=arrays["peaks"],
        intensities=arrays["raw_intensities"],
        stds=arrays["stds"],
        covs=arrays["covs"],
        roi_dataframe=roi_dataframe,
        overwrite=True,
    )
    folder = tmp_path / "Analysis_recording"
    expected_files = {
        "recording_analysis_output.csv",
        "recording_analysis_parameters.csv",
        "recording_stds.csv.zst",
        "recording_peaks.csv.zst",
        "recording_detrended_intensities.csv.zst",
        "recording_raw_intensities.csv.zst",
        "recording_roi_properties.csv",
        "recording_synchronicity_on_tproj.svg",
        "recording_activity_heatmap.svg",
    }
    assert {path.name for path in folder.iterdir()} == expected_files

    for name in ("peaks", "detrended", "raw_intensities"):
        file_name = {
            "peaks": "peaks",
            "detrended": "detrended_intensities",
            "raw_intensities": "raw_intensities",
        }[name]
        saved = read_Txnrois(str(folder / f"recording_{file_name}.csv.zst"))
        np.testing.assert_allclose(saved, arrays[name], rtol=1e-6)

    saved_output = pd.read_csv(folder / "recording_analysis_output.csv").iloc[0]
    assert saved_output["number_of_cells"] == output.number_of_cells
    assert saved_output["normalized_peaks"] == output.normalized_peaks
    saved_params = pd.read_csv(folder / "recording_analysis_parameters.csv").iloc[0]
    assert saved_params["neighbour_tolerance"] == params.neighbour_tolerance
    saved_rois = pd.read_csv(folder / "recording_roi_properties.csv")
    np.testing.assert_allclose(
        saved_rois.to_numpy(dtype=float), arrays["roi_dataframe"], rtol=1e-12
    )


if __name__ == "__main__":
    _, output, arrays, _ = run_analysis()
    del arrays["raw_intensities"], arrays["masks"]
    np.savez_compressed(EXPECTED_ARRAYS, **arrays)
    EXPECTED_OUTPUT.write_text(json.dumps(_output_dict(output), indent=2) + "\n")
    print(f"Saved the expected results to {EXPECTED_ARRAYS} and {EXPECTED_OUTPUT}")
