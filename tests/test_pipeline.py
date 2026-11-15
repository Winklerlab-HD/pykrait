import pytest
import dask.array as da
from pathlib import Path
from unittest.mock import MagicMock, patch
import warnings

import os

import numpy as np
import pandas as pd

from pykrait.pipeline.pipeline import (
    load_timelapse,
    AnalysisParameters,
    AnalysisOutput,
    BatchExperiment,
)

TEST_TIMELAPSE = da.zeros((10, 1, 1, 52, 52))


@patch(
    "pykrait.pipeline.pipeline.load_timelapse_lazy",
    return_value=(TEST_TIMELAPSE, 0.2, 1.23, 4.56),
)
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


@patch(
    "pykrait.pipeline.pipeline.load_timelapse_lazy",
    return_value=(TEST_TIMELAPSE, 0.033333, 1.23, 4.56),
)
def test_meta_and_param_within_tolerance_uses_meta(mock_lazy, tmp_path: Path):
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


@patch(
    "pykrait.pipeline.pipeline.load_timelapse_lazy",
    return_value=(TEST_TIMELAPSE, 0.03, 1.23, 4.56),
)
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


@pytest.mark.parametrize(
    "meta, param, expected, source",
    [
        # a rounded copy of the metadata: the precise value wins
        (1.729, 1.73, 1.729, "metadata"),
        # equal when rounded to 2 decimals, but 4x apart
        (0.004, 0.001, 0.001, "parameter"),
        # 0 in the metadata is not a frame interval
        (0.0, 2.0, 2.0, "parameter"),
        (0.0, None, 1.0, "default"),
        (0.5, None, 0.5, "metadata"),
    ],
)
def test_frame_interval_from_metadata_and_parameters(
    tmp_path, meta, param, expected, source
):
    f = tmp_path / "CalciumVideo.czi"
    f.write_text("stub")
    with (
        patch(
            "pykrait.pipeline.pipeline.load_timelapse_lazy",
            return_value=(TEST_TIMELAPSE, meta, 1.0, 1.0),
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        _, out = load_timelapse(
            AnalysisOutput(filepath=str(f), filename=f.name),
            AnalysisParameters(frame_interval=param),
        )
    assert out.frame_interval == pytest.approx(expected)
    assert out.frame_interval_source == source


@pytest.fixture
def experiment_folder(tmp_path: Path):
    """Folder tree with one subfolder per session and several acquisition types."""
    filenames = [
        "sample1_treated_timeseries.czi",
        "sample1_timeseries.czi",
        "sample1_timeseries_plane.czi",
        "sample1_snap.czi",
        "sample1_zstack.czi",
    ]
    for session in ["session_1", "session_2"]:
        session_folder = tmp_path / "sample1" / session
        session_folder.mkdir(parents=True)
        for filename in filenames:
            (session_folder / filename).write_text("stub")
    return tmp_path


def test_batch_experiment_warns_when_nothing_matches(experiment_folder: Path):
    with pytest.warns(UserWarning, match="No '.czi' files found"):
        experiment = BatchExperiment(
            folder=str(experiment_folder), params=AnalysisParameters()
        )
    assert experiment.experiments == []


def test_batch_experiment_selects_filtered_videos(experiment_folder: Path):
    experiment = BatchExperiment(
        folder=str(experiment_folder),
        params=AnalysisParameters(),
        filter_string="*_timeseries.czi",
        recursive=True,
    )
    selected = sorted(
        os.path.basename(e.analysis_output.filepath) for e in experiment.experiments
    )
    assert selected == [
        "sample1_timeseries.czi",
        "sample1_timeseries.czi",
        "sample1_treated_timeseries.czi",
        "sample1_treated_timeseries.czi",
    ]


def test_batch_experiment_exclude_string(experiment_folder: Path):
    experiment = BatchExperiment(
        folder=str(experiment_folder),
        params=AnalysisParameters(),
        filter_string="timeseries",
        exclude_string="_plane",
        recursive=True,
    )
    assert len(experiment.experiments) == 4
    assert all(
        "_plane" not in e.analysis_output.filepath for e in experiment.experiments
    )


# --- Channel selection for the figures ---


def test_select_intensity_channel_picks_the_non_reference_channel():
    """A two-channel projection must be reduced to the channel the traces come from."""
    import numpy as np
    from pykrait.pipeline.pipeline import select_intensity_channel, INTENSITY_CHANNEL

    projection = np.stack([np.full((8, 8), 10), np.full((8, 8), 20)])  # (C, Y, X)
    selected = select_intensity_channel(projection)

    assert selected.shape == (8, 8)
    assert np.all(selected == 10)  # channel 0, not the reference channel
    assert INTENSITY_CHANNEL == 0


def test_select_intensity_channel_passes_through_2d_and_single_channel():
    import numpy as np
    from pykrait.pipeline.pipeline import select_intensity_channel

    flat = np.zeros((8, 8))
    assert select_intensity_channel(flat).shape == (8, 8)
    assert select_intensity_channel(np.zeros((1, 8, 8))).shape == (8, 8)


def test_select_intensity_channel_falls_back_for_missing_channel():
    import numpy as np
    from pykrait.pipeline.pipeline import select_intensity_channel

    projection = np.stack([np.full((4, 4), 1), np.full((4, 4), 2)])
    with pytest.warns(UserWarning, match="does not exist"):
        selected = select_intensity_channel(projection, channel=5)
    assert np.all(selected == 1)


def test_select_intensity_channel_rejects_4d():
    import numpy as np
    from pykrait.pipeline.pipeline import select_intensity_channel

    with pytest.raises(ValueError):
        select_intensity_channel(np.zeros((2, 2, 8, 8)))


def test_save_analysis_results_accepts_a_multichannel_projection(tmp_path: Path):
    """Regression: a fresh run hands save_analysis_results the (C, Y, X) projection."""
    import numpy as np
    from pykrait.pipeline.pipeline import save_analysis_results

    video = tmp_path / "video.czi"
    video.write_text("stub")

    n_rois, n_frames = 3, 20
    masks = np.zeros((32, 32), dtype=np.uint16)
    for roi in range(n_rois):
        masks[4 + roi * 8 : 10 + roi * 8, 4:10] = roi + 1

    analysis_output = AnalysisOutput(
        filepath=str(video),
        filename=video.name,
        frame_interval=1.0,
        number_of_cells=n_rois,
    )
    peaks = np.zeros((n_frames, n_rois), dtype=int)
    peaks[5] = 1

    save_analysis_results(
        tproj=np.random.default_rng(0)
        .integers(0, 65535, (2, 32, 32))
        .astype(np.uint16),
        masks=masks,
        synchronous_peaks_matrix=np.zeros((n_rois, n_rois)),
        analysis_output=analysis_output,
        analysis_params=AnalysisParameters(),
        detrended_intensities=np.zeros((n_frames, n_rois)),
        peaks=peaks,
        intensities=np.zeros((n_frames, n_rois)),
        stds=np.zeros(n_rois),
        covs=np.zeros(n_rois),
        overwrite=True,
    )

    analysis_folder = tmp_path / "Analysis_video"
    assert (analysis_folder / "video_synchronicity_on_tproj.svg").exists()
    assert (analysis_folder / "video_activity_heatmap.svg").exists()


@patch(
    "pykrait.pipeline.pipeline.load_timelapse_lazy",
    return_value=(TEST_TIMELAPSE[:, :, 0], None, 1.0, 1.0),
)
def test_nan_frame_interval_from_saved_params_counts_as_missing(
    mock_lazy, tmp_path: Path
):
    """A frame_interval of None is saved to CSV as an empty cell and read back as NaN."""
    import numpy as np

    analysis_output = AnalysisOutput(filepath=str(tmp_path / "video.czi"))
    params = AnalysisParameters(frame_interval=np.nan)

    with pytest.warns(UserWarning, match="Using default value of 1.0 seconds"):
        _, out = load_timelapse(analysis_output, params)

    assert out.frame_interval == 1.0
    assert params.frame_interval is None


@patch(
    "pykrait.pipeline.pipeline.load_timelapse_lazy",
    return_value=(TEST_TIMELAPSE[:, :, 0], 0.5, None, None),
)
def test_missing_pixel_size_defaults_to_one_micrometer(mock_lazy, tmp_path: Path):
    analysis_output = AnalysisOutput(filepath=str(tmp_path / "video.tif"))

    with pytest.warns(UserWarning, match="No pixel size"):
        _, out = load_timelapse(analysis_output, AnalysisParameters())

    assert (out.pixel_interval_y, out.pixel_interval_x) == (1.0, 1.0)
    assert out.number_of_channels == 1


def test_analysis_output_from_older_version_loads_with_missing_fields():
    import pandas as pd

    old_output = AnalysisOutput(filepath="video.czi", frame_interval=0.5).to_pandas()
    old_output = old_output.drop(
        columns=["motion_corrected_path", "number_of_channels", "error"]
    )

    loaded = AnalysisOutput.from_pandas(pd.DataFrame(old_output))

    assert loaded.frame_interval == 0.5
    assert loaded.motion_corrected_path is None
    assert loaded.error is None


@pytest.mark.parametrize(
    "intensity_channel, reference_channel, n_original, n_channels, expected",
    [
        (0, 1, 2, 2, 0),  # nothing dropped
        (1, 0, 2, 2, 1),  # nothing dropped, intensities from channel 1
        (1, 0, 2, 1, 0),  # reference 0 dropped, channel 1 moves to index 0
        (0, 1, 2, 1, 0),  # reference 1 dropped, channel 0 keeps its index
        (2, 0, 3, 2, 1),
    ],
)
def test_intensity_channel_index_maps_through_dropped_reference(
    intensity_channel, reference_channel, n_original, n_channels, expected
):
    from pykrait.pipeline.pipeline import intensity_channel_index

    params = AnalysisParameters(
        intensity_channel=intensity_channel,
        motion_correction_channel=reference_channel,
    )
    assert intensity_channel_index(params, n_original, n_channels) == expected


def test_intensity_channel_index_rejects_dropped_or_missing_channel():
    from pykrait.pipeline.pipeline import intensity_channel_index

    with pytest.raises(ValueError, match="reference channel"):
        intensity_channel_index(
            AnalysisParameters(intensity_channel=1, motion_correction_channel=1), 2, 1
        )
    with pytest.raises(ValueError, match="does not exist"):
        intensity_channel_index(AnalysisParameters(intensity_channel=2), 2, 2)


def _save_parameters(folder: Path, params: AnalysisParameters, drop=()):
    frame = params.to_pandas().drop(columns=list(drop))
    frame.to_csv(folder / "video_analysis_parameters.csv", index=False)


def test_saved_parameters_differ_detects_changed_segmentation(tmp_path: Path):
    from pykrait.pipeline.pipeline import (
        _saved_parameters_differ,
        SEGMENTATION_PARAMETERS,
    )

    _save_parameters(tmp_path, AnalysisParameters(tproj_type="std"))

    assert not _saved_parameters_differ(
        str(tmp_path),
        "video",
        AnalysisParameters(tproj_type="std", sinc_filter_window=1),
        SEGMENTATION_PARAMETERS,
        "masks",
    )
    assert _saved_parameters_differ(
        str(tmp_path),
        "video",
        AnalysisParameters(tproj_type="sum"),
        SEGMENTATION_PARAMETERS,
        "masks",
    )
    assert _saved_parameters_differ(
        str(tmp_path),
        "video",
        AnalysisParameters(motion_correction=True),
        SEGMENTATION_PARAMETERS,
        "masks",
    )


def test_saved_parameters_differ_ignores_settings_of_disabled_motion_correction(
    tmp_path: Path,
):
    from pykrait.pipeline.pipeline import (
        _saved_parameters_differ,
        SEGMENTATION_PARAMETERS,
    )

    _save_parameters(tmp_path, AnalysisParameters(motion_correction_grid_size=10))

    assert not _saved_parameters_differ(
        str(tmp_path),
        "video",
        AnalysisParameters(motion_correction_grid_size=5),
        SEGMENTATION_PARAMETERS,
        "masks",
    )


def test_saved_parameters_differ_uses_defaults_for_parameters_of_older_versions(
    tmp_path: Path,
):
    from pykrait.pipeline.pipeline import _saved_parameters_differ, INTENSITY_PARAMETERS

    _save_parameters(tmp_path, AnalysisParameters(), drop=["intensity_channel"])

    assert not _saved_parameters_differ(
        str(tmp_path),
        "video",
        AnalysisParameters(),
        INTENSITY_PARAMETERS,
        "intensities",
    )
    assert _saved_parameters_differ(
        str(tmp_path),
        "video",
        AnalysisParameters(intensity_channel=1),
        INTENSITY_PARAMETERS,
        "intensities",
    )


def test_saved_parameters_differ_reuses_when_nothing_was_saved(tmp_path: Path):
    from pykrait.pipeline.pipeline import (
        _saved_parameters_differ,
        SEGMENTATION_PARAMETERS,
    )

    with pytest.warns(UserWarning, match="cannot be checked"):
        assert not _saved_parameters_differ(
            str(tmp_path),
            "video",
            AnalysisParameters(),
            SEGMENTATION_PARAMETERS,
            "masks",
        )


def test_batch_experiment_gives_every_video_its_own_parameters(experiment_folder: Path):
    params = AnalysisParameters(time_window=3)
    batch = BatchExperiment(
        folder=str(experiment_folder),
        params=params,
        filter_string="timeseries",
        recursive=True,
    )

    batch.experiments[0].analysis_parameters.time_window = 99
    assert params.time_window == 3
    assert batch.experiments[1].analysis_parameters.time_window == 3


def test_batch_experiment_load_params_falls_back_to_provided_params(
    experiment_folder: Path,
):
    batch = BatchExperiment(
        folder=str(experiment_folder / "sample1" / "session_1"),
        params=AnalysisParameters(time_window=3),
        filter_string="sample1_timeseries.czi",
        load_params=True,
    )
    assert batch.experiments[0].analysis_parameters.time_window == 3


def test_batch_experiment_reports_failures_and_does_not_duplicate_rows(
    experiment_folder: Path,
):
    """The stub files cannot be read, so every video fails."""
    batch = BatchExperiment(
        folder=str(experiment_folder),
        params=AnalysisParameters(),
        filter_string="timeseries",
        recursive=True,
    )
    n_videos = len(batch.experiments)

    for _ in range(2):
        with pytest.warns(UserWarning, match=f"failed for {n_videos} of {n_videos}"):
            batch.run()

    assert len(batch.results) == n_videos
    assert batch.results["error"].notna().all()

    with pytest.raises(Exception):
        batch.run(raise_errors=True)


def test_interrupted_motion_correction_leaves_nothing_to_reuse(tmp_path: Path):
    import json
    import numpy as np
    from pykrait.pipeline.pipeline import apply_motion_correction

    analysis_output = AnalysisOutput(
        filepath=str(tmp_path / "video.czi"), analysis_folder=str(tmp_path)
    )
    corrected_path = tmp_path / "video_motion_corrected.tif"
    settings_path = tmp_path / "video_motion_correction.json"
    # a finished correction with other settings
    corrected_path.write_text("old video")
    settings_path.write_text(json.dumps({"motion_correction_method": "global_shift"}))

    def interrupted(timelapse, path, params, **kwargs):
        Path(path).write_text("partial")
        raise KeyboardInterrupt

    with patch("pykrait.pipeline.pipeline._correct_and_save", side_effect=interrupted):
        with pytest.raises(KeyboardInterrupt):
            apply_motion_correction(
                da.zeros((4, 1, 8, 8), dtype=np.uint16),
                analysis_output,
                AnalysisParameters(motion_correction=True),
                verbose=False,
            )

    # the old video keeps its settings, so the next run sees that it was corrected with
    # other settings and corrects again instead of reusing it without settings
    assert corrected_path.read_text() == "old video"
    assert json.loads(settings_path.read_text()) == {
        "motion_correction_method": "global_shift"
    }
    # the partial video of the interrupted correction is deleted
    assert not (tmp_path / "video_motion_corrected.tif.partial.tif").exists()

    def corrected(timelapse, path, params, **kwargs):
        Path(path).write_text("new video")

    rerun = MagicMock(side_effect=corrected)
    with (
        patch("pykrait.pipeline.pipeline._correct_and_save", rerun),
        patch(
            "pykrait.pipeline.pipeline.load_timelapse_lazy",
            return_value=(da.zeros((4, 1, 8, 8)), None, None, None),
        ),
    ):
        apply_motion_correction(
            da.zeros((4, 1, 8, 8), dtype=np.uint16),
            analysis_output,
            AnalysisParameters(motion_correction=True),
            verbose=False,
        )
    rerun.assert_called_once()
    assert corrected_path.read_text() == "new video"


def test_masks_with_gaps_are_relabeled_and_intensities_stay_aligned():
    import numpy as np
    from pykrait.io.images import clean_label_image
    from pykrait.preprocessing.timeseries_extraction import extract_mean_intensities

    masks = np.zeros((40, 40), dtype=np.uint16)
    masks[2:12, 2:12] = 1
    masks[20:30, 20:30] = 3  # label 2 was deleted

    timelapse = np.zeros((3, 1, 40, 40))
    timelapse[:, :, 20:30, 20:30] = 7.0
    timelapse = da.from_array(timelapse)

    # without relabeling, the missing label gets a NaN column instead of shifting ROIs
    with pytest.warns(UserWarning, match=r"Labels \[2\] do not exist"):
        intensities = extract_mean_intensities(timelapse, masks, verbose=False)
    assert intensities.shape == (3, 3)
    assert np.isnan(intensities[:, 1]).all()
    assert np.allclose(intensities[:, 2], 7.0)

    with pytest.warns(UserWarning, match="Relabeling"):
        relabeled = clean_label_image(masks)
    assert sorted(np.unique(relabeled)) == [0, 1, 2]
    intensities = extract_mean_intensities(timelapse, relabeled, verbose=False)
    assert intensities.shape == (3, 2)
    assert np.allclose(intensities[:, 1], 7.0)


@pytest.fixture
def small_video(tmp_path: Path):
    """Two neighbouring cells that fire at the same frames, as a calibrated ImageJ TIFF."""
    import numpy as np
    import tifffile

    n_frames, size = 120, 64
    video = np.full((n_frames, size, size), 100, dtype=np.uint16)
    for frame in range(10, n_frames, 20):
        video[frame : frame + 3, 10:30, 10:30] = 3000
        video[frame : frame + 3, 10:30, 32:52] = 3000
    path = tmp_path / "video.tif"
    tifffile.imwrite(
        path,
        video,
        imagej=True,
        resolution=(1.0, 1.0),
        metadata={"axes": "TYX", "finterval": 1.0, "unit": "um"},
    )

    masks = np.zeros((size, size), dtype=np.uint16)
    masks[10:30, 10:30] = 1
    masks[10:30, 32:52] = 2
    return path, masks


def test_calcium_video_reuses_saved_results_only_while_parameters_match(small_video):
    from pykrait.pipeline.pipeline import CalciumVideo

    path, masks = small_video
    params = AnalysisParameters(
        peak_prominence=500, peak_min_height=50, sinc_filter_window=60
    )
    video = CalciumVideo(str(path), params)

    with patch(
        "pykrait.pipeline.pipeline.create_cellpose_segmentation", return_value=masks
    ) as segmentation:
        first = video.run(raise_errors=True)
        assert first.error is None
        assert first.number_of_cells == 2
        assert first.total_coinciding_peaks_1st_neighbour > 0
        assert segmentation.call_count == 1

        # unchanged parameters: masks and intensities are reused
        second = video.run(
            skip_cellpose=True, skip_intensity_extraction=True, raise_errors=True
        )
        assert segmentation.call_count == 1
        assert second.normalized_peaks == first.normalized_peaks

        # a different projection gives different masks, and new masks new intensities
        params.tproj_type = "sum"
        video.run(skip_cellpose=True, skip_intensity_extraction=True, raise_errors=True)
        assert segmentation.call_count == 2


def test_neighbour_tolerance_in_pixels_uses_x_pixel_size_and_clamps():
    from pykrait.pipeline.pipeline import neighbour_tolerance_in_pixels

    output = AnalysisOutput(pixel_interval_x=0.5, pixel_interval_y=2.0)
    assert (
        neighbour_tolerance_in_pixels(
            AnalysisParameters(neighbour_tolerance=10), output
        )
        == 20
    )
    with pytest.warns(UserWarning, match="too small"):
        assert (
            neighbour_tolerance_in_pixels(
                AnalysisParameters(neighbour_tolerance=0.2), output
            )
            == 1
        )


def test_peak_widths_in_frames_converts_and_clamps():
    from pykrait.pipeline.pipeline import peak_widths_in_frames

    output = AnalysisOutput(frame_interval=2.0)
    assert peak_widths_in_frames(
        AnalysisParameters(peak_min_width=4, peak_max_width=40), output
    ) == (2, 20)
    with pytest.warns(UserWarning):
        assert peak_widths_in_frames(
            AnalysisParameters(peak_min_width=1, peak_max_width=1), output
        ) == (1, 2)


def test_parameters_from_csv_round_trip_exactly(tmp_path: Path):
    from pykrait.pipeline.pipeline import save_params

    params = AnalysisParameters(
        frame_interval=None,
        cov_threshold=0.1 + 0.2,
        peak_max_width=80,
        intensity_channel=1,
    )
    output = AnalysisOutput(
        filepath=str(tmp_path / "v.czi"),
        std_quantile=0.010434782608695651,
        timestamp=np.datetime64("now"),
    )
    save_params(output, params)
    folder = tmp_path / "Analysis_v"

    loaded = AnalysisParameters.from_csv(str(folder / "v_analysis_parameters.csv"))
    assert loaded == params
    assert loaded.cov_threshold == 0.1 + 0.2  # not rounded by the CSV parser
    assert (
        type(loaded.peak_max_width) is float and type(loaded.intensity_channel) is int
    )
    assert loaded.frame_interval is None

    loaded_output = AnalysisOutput.from_csv(str(folder / "v_analysis_output.csv"))
    assert loaded_output.std_quantile == output.std_quantile
    assert loaded_output.timestamp == output.timestamp


def test_parameters_from_csv_warns_about_unknown_and_missing_entries(tmp_path: Path):
    path = tmp_path / "params.csv"
    pd.DataFrame([{"tproj_type": "sum", "not_a_parameter": 1}]).to_csv(
        path, index=False
    )
    with pytest.warns(UserWarning) as record:
        loaded = AnalysisParameters.from_csv(str(path))
    messages = " ".join(str(w.message) for w in record)
    assert (
        "unknown entries ['not_a_parameter']" in messages
        and "No saved value" in messages
    )
    assert (
        loaded.tproj_type == "sum"
        and loaded.peak_min_height == AnalysisParameters().peak_min_height
    )


def test_calcium_video_seed_makes_the_random_controls_reproducible(small_video):
    from pykrait.pipeline.pipeline import CalciumVideo

    path, masks = small_video
    params = AnalysisParameters(
        peak_prominence=500, peak_min_height=50, sinc_filter_window=60
    )
    video = CalciumVideo(str(path), params)

    def comparable(output):
        return output.to_pandas().drop(columns=["timestamp"])

    with patch(
        "pykrait.pipeline.pipeline.create_cellpose_segmentation", return_value=masks
    ):
        first = comparable(video.run(raise_errors=True, seed=7))
        np.random.seed(123)  # the seed has to win over the global state before the run
        second = comparable(
            video.run(
                skip_cellpose=True,
                skip_intensity_extraction=True,
                raise_errors=True,
                seed=7,
            )
        )
    pd.testing.assert_frame_equal(first, second)


def test_calcium_video_results_do_not_depend_on_the_video_cache(small_video, tmp_path):
    import glob
    import tempfile

    from pykrait.pipeline.pipeline import CalciumVideo

    path, masks = small_video
    results = {}
    for mode in ("none", "memory", "disk", "auto"):
        video = CalciumVideo(
            str(path),
            AnalysisParameters(
                peak_prominence=500, peak_min_height=50, sinc_filter_window=60
            ),
        )
        with patch(
            "pykrait.pipeline.pipeline.create_cellpose_segmentation", return_value=masks
        ):
            output = video.run(raise_errors=True, seed=3, video_cache=mode)
        results[mode] = (
            output.to_pandas().drop(columns=["timestamp"]),
            video.mean_intensities,
            video.tproj,
        )
    for mode in ("memory", "disk", "auto"):
        pd.testing.assert_frame_equal(results[mode][0], results["none"][0])
        np.testing.assert_array_equal(results[mode][1], results["none"][1])
        np.testing.assert_array_equal(results[mode][2], results["none"][2])
    assert not glob.glob(os.path.join(tempfile.gettempdir(), "pykrait_frames_*"))


def test_interrupted_segmentation_does_not_leave_masks_to_reuse(small_video):
    from pykrait.pipeline.pipeline import CalciumVideo

    path, masks = small_video
    other_masks = masks.copy()
    other_masks[40:60, 10:30] = 3
    params = AnalysisParameters(
        peak_prominence=500, peak_min_height=50, sinc_filter_window=60
    )
    video = CalciumVideo(str(path), params)
    intensities = path.parent / "Analysis_video" / "video_raw_intensities.csv.zst"

    with patch(
        "pykrait.pipeline.pipeline.create_cellpose_segmentation", return_value=masks
    ) as segmentation:
        assert video.run(raise_errors=True).number_of_cells == 2
    assert intensities.exists()

    # new masks with other parameters, interrupted before the intensities are saved
    params.tproj_type = "sum"
    with (
        patch(
            "pykrait.pipeline.pipeline.create_cellpose_segmentation",
            return_value=other_masks,
        ),
        patch(
            "pykrait.pipeline.pipeline.create_mean_intensities",
            side_effect=RuntimeError("interrupted"),
        ),
        pytest.raises(RuntimeError),
    ):
        video.run(skip_cellpose=True, raise_errors=True)
    # the intensities of the replaced masks are gone
    assert not intensities.exists()

    # the saved masks belong to "sum", so the original parameters segment again
    params.tproj_type = "std"
    with patch(
        "pykrait.pipeline.pipeline.create_cellpose_segmentation", return_value=masks
    ) as segmentation:
        result = video.run(
            skip_cellpose=True, skip_intensity_extraction=True, raise_errors=True
        )
    assert segmentation.call_count == 1
    assert result.number_of_cells == 2


def test_reused_intensities_must_match_the_masks(small_video):
    import numpy as np
    import tifffile
    from pykrait.pipeline.pipeline import CalciumVideo

    path, masks = small_video
    params = AnalysisParameters(
        peak_prominence=500, peak_min_height=50, sinc_filter_window=60
    )
    video = CalciumVideo(str(path), params)
    with patch(
        "pykrait.pipeline.pipeline.create_cellpose_segmentation", return_value=masks
    ):
        video.run(raise_errors=True)

    # a third ROI added to the saved masks by hand
    edited = masks.copy()
    edited[40:60, 10:30] = 3
    masks_path = path.parent / "Analysis_video" / "video_cp_masks.ome.tif"
    tifffile.imwrite(masks_path, edited, metadata={"axes": "YX"})
    with pytest.raises(ValueError, match="Expected 3 ROIs"):
        video.run(skip_cellpose=True, skip_intensity_extraction=True, raise_errors=True)
    assert np.array_equal(video.masks, edited)


@pytest.mark.parametrize("save_temp_file", [True, False])
def test_calcium_video_passes_save_temp_file_to_the_cache(small_video, save_temp_file):
    from pykrait.io.frame_cache import FrameCache
    from pykrait.pipeline.pipeline import CalciumVideo

    path, masks = small_video
    video = CalciumVideo(
        str(path),
        AnalysisParameters(
            peak_prominence=500, peak_min_height=50, sinc_filter_window=60
        ),
    )
    with (
        patch(
            "pykrait.pipeline.pipeline.create_cellpose_segmentation",
            return_value=masks,
        ),
        patch.object(FrameCache, "for_shape", wraps=FrameCache.for_shape) as for_shape,
    ):
        assert video.run(raise_errors=True, save_temp_file=save_temp_file).error is None
    assert for_shape.call_args.kwargs["save_temp_file"] is save_temp_file


def test_calcium_video_without_cells_says_so(small_video):
    import numpy as np
    from pykrait.pipeline.pipeline import CalciumVideo

    path, masks = small_video
    video = CalciumVideo(str(path), AnalysisParameters())
    with patch(
        "pykrait.pipeline.pipeline.create_cellpose_segmentation",
        return_value=np.zeros_like(masks),
    ):
        result = video.run()
    assert result.error.startswith("ValueError: No cells were found in video.tif")


@patch(
    "pykrait.pipeline.pipeline.load_timelapse_lazy",
    return_value=(da.zeros((2, 1, 8, 8)), 1.0, 1.0, 1.0),
)
def test_too_short_video_says_so(mock_lazy, tmp_path: Path):
    f = tmp_path / "snapshot.czi"
    f.write_text("stub")
    with pytest.raises(ValueError, match="snapshot.czi has only 2 frame"):
        load_timelapse(
            AnalysisOutput(filepath=str(f), filename=f.name), AnalysisParameters()
        )


@pytest.mark.parametrize("frame_interval", [None, 2.0])
def test_uncalibrated_video_runs_and_its_corrected_video_stays_uncalibrated(
    tmp_path: Path, frame_interval
):
    import numpy as np
    import tifffile
    from pykrait.io.images import load_timelapse_lazy
    from pykrait.pipeline.pipeline import CalciumVideo

    # a plain TIFF without frame interval or pixel size
    n_frames, size = 60, 64
    video = np.full((n_frames, size, size), 100, dtype=np.uint16)
    for frame in range(10, n_frames, 20):
        video[frame : frame + 3, 10:30, 10:30] = 3000
        video[frame : frame + 3, 10:30, 32:52] = 3000
    path = tmp_path / "plain.tif"
    tifffile.imwrite(path, video)
    masks = np.zeros((size, size), dtype=np.uint16)
    masks[10:30, 10:30] = 1
    masks[10:30, 32:52] = 2

    params = AnalysisParameters(
        peak_prominence=500,
        peak_min_height=50,
        sinc_filter_window=30,
        frame_interval=frame_interval,
        motion_correction=True,
        motion_correction_method="global_shift",
    )
    with (
        patch(
            "pykrait.pipeline.pipeline.create_cellpose_segmentation",
            return_value=masks,
        ),
        warnings.catch_warnings(),
    ):
        warnings.simplefilter("ignore")
        result = CalciumVideo(str(path), params).run(raise_errors=True)

    # the analysis runs on the fallback values and says where they come from
    assert result.number_of_cells == 2
    assert result.pixel_size_source == "default"
    assert result.frame_interval_source == (
        "default" if frame_interval is None else "parameter"
    )

    # the corrected video only carries calibration the original or the user gave
    with tifffile.TiffFile(result.motion_corrected_path) as tif:
        metadata = tif.imagej_metadata
    assert "unit" not in metadata
    assert metadata.get("finterval") == frame_interval
    _, read_interval, y_um, x_um = load_timelapse_lazy(result.motion_corrected_path)
    assert (y_um, x_um) == (None, None)
    assert read_interval == frame_interval
