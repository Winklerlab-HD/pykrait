import json
import pytest
import numpy as np
import dask.array as da
import tifffile
from pathlib import Path
from scipy.ndimage import shift as ndimage_shift

from pykrait.preprocessing.motion_correction import (
    MOTION_CORRECTION_METHODS,
    _to_uint8,
    _fit_polynomial_surface,
    _smooth_flow_on_grid,
    compute_template,
    correct_lateral_motion,
)
from pykrait.pipeline.pipeline import (
    AnalysisParameters,
    AnalysisOutput,
    apply_motion_correction,
)

# --- Fixtures / helpers ---

FRAME_SIZE = 96
SHIFTS = [(0, 0), (3, -2), (-4, 5), (2, 2), (-1, -3)]


def _blob_image(seed: int, size: int = FRAME_SIZE) -> np.ndarray:
    """Returns a synthetic image with a handful of Gaussian blobs."""
    rows, cols = np.mgrid[0:size, 0:size]
    rng = np.random.default_rng(seed)
    image = np.zeros((size, size), dtype=np.float64)
    for _ in range(12):
        center_row = rng.uniform(15, size - 15)
        center_col = rng.uniform(15, size - 15)
        image += 3000 * np.exp(
            -((rows - center_row) ** 2 + (cols - center_col) ** 2) / (2 * 4.0**2)
        )
    return image


@pytest.fixture
def shifted_timelapse():
    """Returns a (T, C, Y, X) = (5, 2, 96, 96) uint16 stack in which both channels are
    translated by the same known offsets."""
    channels = [_blob_image(seed=1), _blob_image(seed=2)]
    frames = [
        np.stack(
            [
                ndimage_shift(channel, offset, mode="constant", cval=0)
                for channel in channels
            ]
        )
        for offset in SHIFTS
    ]
    return np.stack(frames).astype(np.uint16)


@pytest.fixture
def deformed_timelapse():
    """Returns a (T, C, Y, X) = (4, 1, 96, 96) uint16 stack with a smooth non-rigid
    deformation that a rigid translation cannot correct."""
    import cv2

    base = _blob_image(seed=3)
    rows, cols = np.mgrid[0:FRAME_SIZE, 0:FRAME_SIZE]

    frames = []
    for amplitude in [0.0, 2.5, -3.0, 4.0]:
        map_x = (cols + amplitude * np.sin(2 * np.pi * rows / FRAME_SIZE)).astype(
            np.float32
        )
        map_y = (rows + amplitude * np.cos(2 * np.pi * cols / FRAME_SIZE)).astype(
            np.float32
        )
        deformed = cv2.remap(
            base.astype(np.float32), map_x, map_y, interpolation=cv2.INTER_LINEAR
        )
        frames.append(deformed[np.newaxis, ...])
    return np.stack(frames).astype(np.uint16)


def _mean_deviation_from_first(timelapse: np.ndarray, channel: int = 0) -> float:
    """Mean absolute deviation of every frame from the first frame of a channel."""
    reference = timelapse[0, channel].astype(float)
    return float(
        np.mean(
            [
                np.abs(timelapse[t, channel].astype(float) - reference).mean()
                for t in range(timelapse.shape[0])
            ]
        )
    )


# --- Helper function tests ---


def test_to_uint8_maps_quantiles_to_full_range():
    image = np.linspace(0, 1000, 100 * 100).reshape(100, 100)
    result = _to_uint8(image, min_quantile=0.0, max_quantile=1.0)
    assert result.dtype == np.uint8
    assert result.min() == 0
    assert result.max() == 255


def test_to_uint8_clips_outliers():
    image = np.zeros((10, 10))
    image[0, 0] = 10_000  # single outlier that must not compress the rest
    result = _to_uint8(image, min_quantile=0.01, max_quantile=0.99)
    assert result[0, 0] == 255
    assert result[5, 5] == 0


def test_to_uint8_flat_image_returns_zeros():
    result = _to_uint8(np.full((8, 8), 42.0))
    assert result.dtype == np.uint8
    assert np.all(result == 0)


def test_polynomial_surface_recovers_plane():
    rows, cols = np.mgrid[0:20, 0:30]
    surface = 2.0 + 3.0 * cols / 29 - 1.5 * rows / 19
    fitted = _fit_polynomial_surface(surface, degree=1)
    assert fitted.shape == surface.shape
    assert np.allclose(fitted, surface, atol=1e-6)


def test_polynomial_surface_ignores_masked_outliers():
    rows, cols = np.mgrid[0:20, 0:20]
    surface = 1.0 + 2.0 * cols / 19
    corrupted = surface.copy()
    corrupted[:5, :5] = 500.0

    mask = np.ones_like(surface, dtype=np.uint8)
    mask[:5, :5] = 0

    fitted = _fit_polynomial_surface(corrupted, degree=1, mask=mask)
    assert np.allclose(fitted, surface, atol=1e-6)


def test_smooth_flow_on_grid_preserves_constant_flow():
    flow = np.zeros((40, 40, 2), dtype=np.float32)
    flow[..., 0] = 1.5
    flow[..., 1] = -2.5

    smoothed = _smooth_flow_on_grid(flow, grid_size=5)
    assert smoothed.shape == flow.shape
    assert np.allclose(smoothed[..., 0], 1.5, atol=1e-3)
    assert np.allclose(smoothed[..., 1], -2.5, atol=1e-3)


def test_compute_template_averages_first_frames(shifted_timelapse):
    template = compute_template(
        shifted_timelapse, reference_channel=1, template_frames=3
    )
    assert template.dtype == np.float32
    assert template.shape == shifted_timelapse.shape[2:]
    assert np.allclose(
        template, shifted_timelapse[:3, 1].mean(axis=0), rtol=1e-5, atol=1e-3
    )


def test_compute_template_rejects_invalid_channel(shifted_timelapse):
    with pytest.raises(ValueError):
        compute_template(shifted_timelapse, reference_channel=7)


# --- Correction tests ---


@pytest.mark.parametrize("method", MOTION_CORRECTION_METHODS)
def test_correction_removes_translation(shifted_timelapse, method):
    corrected, _ = correct_lateral_motion(
        shifted_timelapse,
        method=method,
        template_frames=1,
        average_frames=1,
        grid_size=6,
        verbose=False,
    )
    before = _mean_deviation_from_first(shifted_timelapse)
    after = _mean_deviation_from_first(corrected)
    assert before > 100  # the synthetic motion is substantial
    assert after < 0.02 * before


@pytest.mark.parametrize("method", MOTION_CORRECTION_METHODS)
def test_correction_preserves_shape_and_dtype(shifted_timelapse, method):
    corrected, template = correct_lateral_motion(
        shifted_timelapse,
        method=method,
        template_frames=2,
        average_frames=3,
        grid_size=6,
        verbose=False,
    )
    assert corrected.shape == shifted_timelapse.shape
    assert corrected.dtype == shifted_timelapse.dtype
    assert template.shape == shifted_timelapse.shape[2:]


def test_correction_applies_transform_to_all_channels(shifted_timelapse):
    """The motion is estimated on channel 0 only, but channel 1 must be corrected too."""
    corrected, _ = correct_lateral_motion(
        shifted_timelapse,
        method="global_shift",
        reference_channel=0,
        template_frames=1,
        average_frames=1,
        verbose=False,
    )
    before = _mean_deviation_from_first(shifted_timelapse, channel=1)
    after = _mean_deviation_from_first(corrected, channel=1)
    assert after < 0.02 * before


def test_correction_works_on_second_reference_channel(shifted_timelapse):
    corrected, _ = correct_lateral_motion(
        shifted_timelapse,
        method="global_shift",
        reference_channel=1,
        template_frames=1,
        average_frames=1,
        verbose=False,
    )
    after = _mean_deviation_from_first(corrected, channel=0)
    assert after < 0.02 * _mean_deviation_from_first(shifted_timelapse, channel=0)


def test_warp_grid_beats_global_shift_on_deformation(deformed_timelapse):
    """A rigid translation cannot correct a non-rigid deformation, the mesh warp can."""
    rigid, _ = correct_lateral_motion(
        deformed_timelapse,
        method="global_shift",
        reference_channel=0,
        template_frames=1,
        average_frames=1,
        verbose=False,
    )
    warped, _ = correct_lateral_motion(
        deformed_timelapse,
        method="warp_grid",
        reference_channel=0,
        template_frames=1,
        average_frames=1,
        grid_size=8,
        verbose=False,
    )
    assert _mean_deviation_from_first(warped) < 0.5 * _mean_deviation_from_first(rigid)


def test_correction_accepts_dask_input(shifted_timelapse):
    lazy_timelapse = da.from_array(
        shifted_timelapse, chunks=(1, 1, FRAME_SIZE, FRAME_SIZE)
    )
    corrected, _ = correct_lateral_motion(
        lazy_timelapse,
        method="global_shift",
        template_frames=1,
        average_frames=1,
        verbose=False,
    )
    assert isinstance(corrected, np.ndarray)
    assert corrected.shape == shifted_timelapse.shape
    assert _mean_deviation_from_first(corrected) < 0.02 * _mean_deviation_from_first(
        shifted_timelapse
    )


def test_correction_accepts_external_template(shifted_timelapse):
    template = compute_template(shifted_timelapse, template_frames=1)
    corrected, returned_template = correct_lateral_motion(
        shifted_timelapse,
        method="global_shift",
        average_frames=1,
        template=template,
        verbose=False,
    )
    # without a template filter the template must not change
    assert np.allclose(returned_template, template)
    assert _mean_deviation_from_first(corrected) < 0.02 * _mean_deviation_from_first(
        shifted_timelapse
    )


def test_template_filter_updates_template(shifted_timelapse):
    """A dimmed template must be pulled back towards the intensities of the recording."""
    true_template = compute_template(shifted_timelapse, template_frames=1)
    dimmed_template = 0.5 * true_template

    _, updated_template = correct_lateral_motion(
        shifted_timelapse,
        method="global_shift",
        average_frames=1,
        template_filter=0.5,
        template=dimmed_template,
        verbose=False,
    )

    assert not np.allclose(updated_template, dimmed_template)
    assert (
        np.abs(updated_template - true_template).mean()
        < np.abs(dimmed_template - true_template).mean()
    )
    # the input template must not be modified in place
    assert np.allclose(dimmed_template, 0.5 * true_template)


def test_template_filter_zero_keeps_template_fixed(deformed_timelapse):
    template = compute_template(
        deformed_timelapse, reference_channel=0, template_frames=1
    )
    _, returned_template = correct_lateral_motion(
        deformed_timelapse,
        method="warp_grid",
        reference_channel=0,
        average_frames=1,
        grid_size=6,
        template_filter=0.0,
        template=template,
        verbose=False,
    )
    assert np.array_equal(returned_template, template)


def test_default_reference_channel_is_the_second_channel(shifted_timelapse):
    """The default estimates the motion on channel 1, which carries the stationary label."""
    default_template = compute_template(shifted_timelapse, template_frames=1)
    assert np.allclose(default_template, shifted_timelapse[0, 1], atol=1e-3)


def test_single_channel_video_falls_back_to_channel_zero(deformed_timelapse):
    """Channel 1 does not exist in a single-channel video, so channel 0 is used."""
    assert deformed_timelapse.shape[1] == 1

    with pytest.warns(UserWarning, match="single-channel video"):
        corrected, _ = correct_lateral_motion(
            deformed_timelapse,
            method="global_shift",
            template_frames=1,
            average_frames=1,
            verbose=False,
        )
    assert corrected.shape == deformed_timelapse.shape

    with pytest.warns(UserWarning, match="single-channel video"):
        template = compute_template(deformed_timelapse, template_frames=1)
    assert np.allclose(template, deformed_timelapse[0, 0], atol=1e-3)


def test_out_of_bounds_channel_still_raises_for_multichannel(shifted_timelapse):
    with pytest.raises(ValueError):
        correct_lateral_motion(shifted_timelapse, reference_channel=5, verbose=False)


def test_mask_valid_fov_zeroes_pixels_that_left_the_field_of_view(shifted_timelapse):
    corrected, _ = correct_lateral_motion(
        shifted_timelapse,
        method="global_shift",
        template_frames=1,
        average_frames=1,
        mask_valid_fov=True,
        verbose=False,
    )
    # the largest shift is 5 pixels, so all border pixels must have been masked out
    assert np.all(corrected[:, :, 0, :] == 0)
    assert np.all(corrected[:, :, :, 0] == 0)
    # the centre of the field of view is never masked
    assert corrected[:, :, FRAME_SIZE // 2, FRAME_SIZE // 2].max() > 0


def test_template_frames_larger_than_timelapse_is_clipped(shifted_timelapse):
    corrected, _ = correct_lateral_motion(
        shifted_timelapse,
        method="global_shift",
        template_frames=1000,
        average_frames=1,
        verbose=False,
    )
    assert corrected.shape == shifted_timelapse.shape


# --- Validation tests ---


def test_rejects_non_array_input():
    with pytest.raises(TypeError):
        correct_lateral_motion([[1, 2], [3, 4]], verbose=False)


def test_rejects_non_4d_input():
    with pytest.raises(ValueError):
        correct_lateral_motion(np.zeros((5, 10, 10), dtype=np.uint16), verbose=False)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"method": "not_a_method"},
        {"reference_channel": 5},
        {"average_frames": 0},
        {"template_filter": 1.5},
        {"method": "warp_grid", "grid_size": 1},
        {"method": "warp_polynomial_fit", "polynomial_degree": 0},
    ],
)
def test_rejects_invalid_parameters(shifted_timelapse, kwargs):
    with pytest.raises(ValueError):
        correct_lateral_motion(shifted_timelapse, verbose=False, **kwargs)


def test_rejects_mismatching_template(shifted_timelapse):
    with pytest.raises(ValueError):
        correct_lateral_motion(
            shifted_timelapse, template=np.zeros((10, 10)), verbose=False
        )


# --- Pipeline integration tests ---


def test_apply_motion_correction_is_a_noop_when_disabled(
    shifted_timelapse, tmp_path: Path
):
    lazy_timelapse = da.from_array(
        shifted_timelapse, chunks=(1, 1, FRAME_SIZE, FRAME_SIZE)
    )
    analysis_output = AnalysisOutput(
        filepath=str(tmp_path / "video.czi"),
        filename="video.czi",
        analysis_folder=str(tmp_path),
    )

    result, output, raw_projection = apply_motion_correction(
        lazy_timelapse, analysis_output, AnalysisParameters()
    )

    assert result is lazy_timelapse
    assert output.motion_corrected_path is None
    assert raw_projection is None
    assert list(tmp_path.iterdir()) == []


def test_apply_motion_correction_saves_and_reuses_corrected_video(
    shifted_timelapse, tmp_path: Path
):
    lazy_timelapse = da.from_array(
        shifted_timelapse, chunks=(1, 1, FRAME_SIZE, FRAME_SIZE)
    )
    analysis_output = AnalysisOutput(
        filepath=str(tmp_path / "video.czi"),
        filename="video.czi",
        analysis_folder=str(tmp_path),
    )
    params = AnalysisParameters(
        motion_correction=True,
        motion_correction_method="global_shift",
        motion_correction_template_frames=1,
        motion_correction_average_frames=1,
    )

    corrected, output, raw_projection = apply_motion_correction(
        lazy_timelapse, analysis_output, params, verbose=False
    )

    # the projection is accumulated while correcting, so the video is not read again
    assert raw_projection is not None
    assert raw_projection.shape == shifted_timelapse.shape[1:]

    expected_path = tmp_path / "video_motion_corrected.tif"
    assert output.motion_corrected_path == str(expected_path)
    assert expected_path.exists()
    assert corrected.shape == shifted_timelapse.shape
    assert _mean_deviation_from_first(
        np.asarray(corrected)
    ) < 0.02 * _mean_deviation_from_first(shifted_timelapse)

    saved = tifffile.imread(expected_path)
    assert saved.shape == shifted_timelapse.shape
    assert saved.dtype == shifted_timelapse.dtype

    # a second call must reuse the file instead of correcting again
    reloaded, output, reused_projection = apply_motion_correction(
        lazy_timelapse, analysis_output, params, verbose=False
    )
    assert (
        reused_projection is None
    )  # nothing was corrected, so nothing was accumulated
    assert reloaded.shape == shifted_timelapse.shape
    assert np.array_equal(np.asarray(reloaded), saved)


def test_apply_motion_correction_recomputes_when_settings_change(
    shifted_timelapse, tmp_path: Path
):
    lazy_timelapse = da.from_array(
        shifted_timelapse, chunks=(1, 1, FRAME_SIZE, FRAME_SIZE)
    )
    analysis_output = AnalysisOutput(
        filepath=str(tmp_path / "video.czi"),
        filename="video.czi",
        analysis_folder=str(tmp_path),
    )
    params = AnalysisParameters(
        motion_correction=True,
        motion_correction_method="global_shift",
        motion_correction_template_frames=1,
        motion_correction_average_frames=1,
    )
    apply_motion_correction(lazy_timelapse, analysis_output, params, verbose=False)

    settings_path = tmp_path / "video_motion_correction.json"
    assert settings_path.exists()
    first_run = tifffile.imread(tmp_path / "video_motion_corrected.tif")

    # masking the field of view must not be served from the cached file
    params.motion_correction_mask_fov = True
    corrected, _, _ = apply_motion_correction(
        lazy_timelapse, analysis_output, params, verbose=False
    )
    corrected = np.asarray(corrected)

    assert not np.array_equal(corrected, first_run)
    assert np.all(corrected[:, :, 0, :] == 0)
    assert json.loads(settings_path.read_text())["motion_correction_mask_fov"] is True


def test_apply_motion_correction_warns_when_settings_are_missing(
    shifted_timelapse, tmp_path: Path
):
    lazy_timelapse = da.from_array(
        shifted_timelapse, chunks=(1, 1, FRAME_SIZE, FRAME_SIZE)
    )
    analysis_output = AnalysisOutput(
        filepath=str(tmp_path / "video.czi"),
        filename="video.czi",
        analysis_folder=str(tmp_path),
    )
    params = AnalysisParameters(
        motion_correction=True,
        motion_correction_method="global_shift",
        motion_correction_template_frames=1,
        motion_correction_average_frames=1,
    )
    apply_motion_correction(lazy_timelapse, analysis_output, params, verbose=False)
    (tmp_path / "video_motion_correction.json").unlink()

    with pytest.warns(UserWarning, match="No motion correction settings"):
        corrected, _, _ = apply_motion_correction(
            lazy_timelapse, analysis_output, params, verbose=False
        )
    assert corrected.shape == shifted_timelapse.shape


def test_apply_motion_correction_without_analysis_folder(
    shifted_timelapse, tmp_path: Path
):
    lazy_timelapse = da.from_array(
        shifted_timelapse, chunks=(1, 1, FRAME_SIZE, FRAME_SIZE)
    )
    analysis_output = AnalysisOutput(
        filepath=str(tmp_path / "video.czi"), filename="video.czi"
    )
    params = AnalysisParameters(motion_correction=True)

    with pytest.raises(ValueError):
        apply_motion_correction(lazy_timelapse, analysis_output, params, verbose=False)


@pytest.mark.parametrize("tproj_type", ["std", "sum"])
def test_motion_correction_accumulates_the_configured_projection(
    shifted_timelapse, tmp_path: Path, tproj_type
):
    """Both projection types are accumulated while correcting, so the corrected video
    never has to be read back to segment it."""
    lazy_timelapse = da.from_array(
        shifted_timelapse, chunks=(1, 1, FRAME_SIZE, FRAME_SIZE)
    )
    analysis_output = AnalysisOutput(
        filepath=str(tmp_path / "video.czi"),
        filename="video.czi",
        analysis_folder=str(tmp_path),
    )
    params = AnalysisParameters(
        tproj_type=tproj_type,
        motion_correction=True,
        motion_correction_method="global_shift",
        motion_correction_template_frames=1,
        motion_correction_average_frames=1,
    )

    _, _, raw_projection = apply_motion_correction(
        lazy_timelapse, analysis_output, params, verbose=False
    )

    corrected = tifffile.imread(tmp_path / "video_motion_corrected.tif")
    expected = (
        corrected.astype(np.float64).sum(axis=0)
        if tproj_type == "sum"
        else corrected.astype(np.float64).std(axis=0)
    )

    # pixels that left the field of view are zeroed in the projection, the rest matches
    # the projection of the saved video
    assert raw_projection.shape == shifted_timelapse.shape[1:]
    differs = ~np.isclose(raw_projection, expected, atol=1e-6)
    assert np.all(raw_projection[differs] == 0)
    assert np.all(corrected.min(axis=0)[differs] == 0)


def _motion_correction_setup(tmp_path: Path, shifted_timelapse, **params):
    lazy_timelapse = da.from_array(
        shifted_timelapse, chunks=(1, 1, FRAME_SIZE, FRAME_SIZE)
    )
    analysis_output = AnalysisOutput(
        filepath=str(tmp_path / "video.czi"),
        filename="video.czi",
        analysis_folder=str(tmp_path),
        number_of_channels=shifted_timelapse.shape[1],
        frame_interval=0.5,
        pixel_interval_y=0.25,
        pixel_interval_x=0.25,
    )
    params = AnalysisParameters(
        motion_correction=True,
        motion_correction_method="global_shift",
        motion_correction_template_frames=1,
        motion_correction_average_frames=1,
        **params,
    )
    return lazy_timelapse, analysis_output, params


def test_corrected_video_is_an_imagej_tiff_with_the_original_metadata(
    shifted_timelapse, tmp_path: Path
):
    """The corrected video is saved as an ImageJ hyperstack, which readers open from its
    first page instead of walking the header of every page."""
    from pykrait.io.images import load_timelapse_lazy

    lazy_timelapse, analysis_output, params = _motion_correction_setup(
        tmp_path, shifted_timelapse
    )
    corrected, output, _ = apply_motion_correction(
        lazy_timelapse, analysis_output, params, verbose=False
    )

    with tifffile.TiffFile(output.motion_corrected_path) as tiff:
        assert tiff.is_imagej
        assert tiff.pages[0].compression == tifffile.COMPRESSION.NONE
        assert tiff.series[0].shape == shifted_timelapse.shape
    _, frame_interval, y_um, x_um = load_timelapse_lazy(output.motion_corrected_path)
    assert frame_interval == pytest.approx(0.5)
    assert (y_um, x_um) == (0.25, 0.25)
    assert np.array_equal(
        np.asarray(corrected), tifffile.imread(output.motion_corrected_path)
    )


def test_large_corrected_videos_are_written_with_the_first_page_only(
    shifted_timelapse, tmp_path: Path, monkeypatch
):
    """Classic TIFF cannot address pages beyond 4 GB, so larger ImageJ files keep only the
    first page, which still describes the whole video."""
    import pykrait.pipeline.pipeline as pipeline

    monkeypatch.setattr(pipeline, "_IMAGEJ_MAX_FULL_SIZE", 1)
    lazy_timelapse, analysis_output, params = _motion_correction_setup(
        tmp_path, shifted_timelapse
    )
    corrected, output, _ = apply_motion_correction(
        lazy_timelapse, analysis_output, params, verbose=False
    )

    with tifffile.TiffFile(output.motion_corrected_path) as tiff:
        assert len(tiff.pages) == 1
    assert corrected.shape == shifted_timelapse.shape
    assert np.array_equal(
        np.asarray(corrected), tifffile.imread(output.motion_corrected_path)
    )


@pytest.mark.parametrize("mask_fov", [False, True])
@pytest.mark.parametrize("drop_reference_channel", [False, True])
def test_motion_correction_fills_the_intensity_cache(
    shifted_timelapse, tmp_path: Path, mask_fov, drop_reference_channel
):
    """The intensity channel is kept while correcting, so the intensities do not read the
    corrected video back from the drive."""
    from pykrait.pipeline.pipeline import create_intensity_cache

    lazy_timelapse, analysis_output, params = _motion_correction_setup(
        tmp_path,
        shifted_timelapse,
        intensity_channel=0,
        tproj_type="sum",  # the shifts are corrected exactly, so the std would be ~0
        motion_correction_channel=1,
        motion_correction_mask_fov=mask_fov,
        motion_correction_drop_reference_channel=drop_reference_channel,
    )
    cache = create_intensity_cache(
        lazy_timelapse, analysis_output, params, mode="memory"
    )
    try:
        corrected, _, raw_projection = apply_motion_correction(
            lazy_timelapse, analysis_output, params, verbose=False, frame_cache=cache
        )
        corrected = np.asarray(corrected)
        assert corrected.shape[1] == (1 if drop_reference_channel else 2)
        assert cache.shape == corrected.shape
        assert np.array_equal(cache.frames, corrected[:, 0])

        # the projection of the intensity channel is that of the saved video, also when
        # the field of view is masked; unmasked, pixels that left it are 0
        expected = corrected[:, 0].astype(np.float64).sum(axis=0)
        assert raw_projection is not None
        if mask_fov:
            assert np.all(corrected[:, :, 0, :] == 0)
            assert np.allclose(raw_projection[0], expected, atol=1e-6)
        else:
            differs = ~np.isclose(raw_projection[0], expected, atol=1e-6)
            assert np.all(raw_projection[0][differs] == 0)
            assert np.all(corrected[:, 0].min(axis=0)[differs] == 0)
    finally:
        cache.close()


def test_reused_corrected_video_leaves_the_cache_empty(
    shifted_timelapse, tmp_path: Path
):
    from pykrait.pipeline.pipeline import create_intensity_cache

    lazy_timelapse, analysis_output, params = _motion_correction_setup(
        tmp_path, shifted_timelapse
    )
    apply_motion_correction(lazy_timelapse, analysis_output, params, verbose=False)
    cache = create_intensity_cache(
        lazy_timelapse, analysis_output, params, mode="memory"
    )
    cache.frames[:] = 7
    try:
        _, _, raw_projection = apply_motion_correction(
            lazy_timelapse, analysis_output, params, verbose=False, frame_cache=cache
        )
        # nothing was corrected, the caller has to fill the cache from the projection
        assert raw_projection is None
        assert np.all(cache.frames == 7)
    finally:
        cache.close()


def test_legacy_ome_tiff_is_reused_and_replaced_when_settings_change(
    shifted_timelapse, tmp_path: Path
):
    """Corrected OME-TIFF files of earlier versions are still reused; correcting again
    replaces them with the ImageJ TIFF."""
    import pykrait.pipeline.pipeline as pipeline

    lazy_timelapse, analysis_output, params = _motion_correction_setup(
        tmp_path, shifted_timelapse
    )
    legacy_path = tmp_path / "video_motion_corrected.ome.tif"
    tifffile.imwrite(
        legacy_path,
        shifted_timelapse,
        ome=True,
        compression="zstd",
        metadata={"axes": "TCYX"},
    )
    (tmp_path / "video_motion_correction.json").write_text(
        json.dumps(pipeline._motion_correction_settings(params))
    )

    reused, output, raw_projection = apply_motion_correction(
        lazy_timelapse, analysis_output, params, verbose=False
    )
    assert raw_projection is None
    assert output.motion_corrected_path == str(legacy_path)
    assert np.array_equal(np.asarray(reused), shifted_timelapse)

    params.motion_correction_method = "warp_grid"
    _, output, raw_projection = apply_motion_correction(
        lazy_timelapse, analysis_output, params, verbose=False
    )
    assert raw_projection is not None
    assert output.motion_corrected_path == str(tmp_path / "video_motion_corrected.tif")
    assert not legacy_path.exists()


@pytest.mark.parametrize("average_frames", [1, 2, 3, 4, 5])
def test_rolling_window_averages_exactly_average_frames(average_frames):
    from pykrait.preprocessing.motion_correction import LateralMotionCorrector

    # every frame's reference channel holds its index, so the average reveals the window
    n_frames = 20
    timelapse = np.broadcast_to(
        np.arange(n_frames, dtype=np.float32)[:, None, None, None], (n_frames, 1, 8, 8)
    ).copy()
    corrector = LateralMotionCorrector(
        timelapse, reference_channel=0, average_frames=average_frames
    )
    used = []
    get_frame = corrector._get_frame
    corrector._get_frame = lambda index: used.append(index) or get_frame(index)
    corrector._reference_gray(10)
    assert len(used) == average_frames
    assert 10 in used
    assert used == list(range(used[0], used[0] + average_frames))
