import pytest
import dask.array as da
from pathlib import Path
from unittest.mock import patch
import warnings

from pykrait.pipeline.pipeline import load_timelapse, AnalysisParameters, AnalysisOutput

TEST_TIMELAPSE = da.zeros((10, 1, 1, 52, 52))

@patch("pykrait.pipeline.pipeline.load_timelapse_lazy",
       return_value=(TEST_TIMELAPSE, 0.2, 1.23, 4.56))
def test_meta_provided_no_param_uses_meta_no_warning(mock_lazy, tmp_path: Path):
    f = tmp_path / "CalciumVideo.czi"
    f.write_text("stub")
    analysis_output = AnalysisOutput(filepath=str(f), filename=f.name)
    params = AnalysisParameters(frame_interval=None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        arr, out = load_timelapse(analysis_output, params)

    mock_lazy.assert_called_once_with(file_path=analysis_output.filepath)
    assert arr is TEST_TIMELAPSE
    assert out.frame_interval == pytest.approx(0.2)
    assert out.pixel_interval_y == 1.23
    assert out.pixel_interval_x == 4.56

@patch("pykrait.pipeline.pipeline.load_timelapse_lazy",
       return_value=(TEST_TIMELAPSE, 0.033333, 1.23, 4.56))
def test_meta_and_param_match_when_rounded_to_2dp_uses_meta(mock_lazy, tmp_path: Path):
    f = tmp_path / "CalciumVideo.czi"
    f.write_text("stub")
    analysis_output = AnalysisOutput(filepath=str(f), filename=f.name)
    params = AnalysisParameters(frame_interval=0.033334)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        arr, out = load_timelapse(analysis_output, params)

    mock_lazy.assert_called_once_with(file_path=analysis_output.filepath)
    assert arr is TEST_TIMELAPSE
    assert out.frame_interval == pytest.approx(0.033333)
    assert out.pixel_interval_y == 1.23
    assert out.pixel_interval_x == 4.56

@patch("pykrait.pipeline.pipeline.load_timelapse_lazy",
       return_value=(TEST_TIMELAPSE, 0.03, 1.23, 4.56))
def test_meta_and_param_mismatch_warns_and_prefers_param(mock_lazy, tmp_path: Path):
    f = tmp_path / "CalciumVideo.czi"
    f.write_text("stub")

    analysis_output = AnalysisOutput(filepath=str(f), filename=f.name)
    params = AnalysisParameters(frame_interval=0.04)

    with pytest.warns(UserWarning) as rec:
        arr, out = load_timelapse(analysis_output, params)

    mock_lazy.assert_called_once_with(file_path=analysis_output.filepath)
    msgs = [str(w.message) for w in rec]
    assert any("does not match inferred frame_interval" in m for m in msgs)

    assert arr is TEST_TIMELAPSE
    assert out is analysis_output
    # provided value should win
    assert out.frame_interval == pytest.approx(0.04)
    assert out.pixel_interval_y == 1.23
    assert out.pixel_interval_x == 4.56

