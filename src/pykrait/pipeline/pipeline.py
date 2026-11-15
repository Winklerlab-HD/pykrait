import logging
import os
import json
import math
import traceback
import tifffile
from tqdm import tqdm
import warnings
import pandas as pd
import numpy as np
import dask.array as da
from dataclasses import dataclass, fields, asdict, replace

from typing import Literal, Tuple

from pykrait.io.images import (
    load_timelapse_lazy,
    get_pixel_contour_from_label_img,
    read_projection,
    read_label_image,
    clean_label_image,
    release_tiff_handles,
)
from pykrait.io.files import (
    get_files_from_folder,
    get_analysis_folder,
    get_analysis_file,
    get_analysis_stem,
    get_pykrait_version,
    save_Txnrois,
    save_NroisxF,
    read_Txnrois,
)
from pykrait.io.frame_cache import FrameCache, VideoCacheMode
from pykrait.preprocessing.segmentation import (
    create_cellpose_segmentation,
    timelapse_projection,
    finalize_projection,
    frames_per_block,
    TemporalProjection,
)
from pykrait.preprocessing.motion_correction import (
    LateralMotionCorrector,
    _resolve_reference_channel,
)
from pykrait.preprocessing.timeseries_extraction import (
    extract_mean_intensities,
    extract_cell_properties,
    get_adjacency_matrix,
)
from pykrait.trace_analysis.filtering import MIN_FRAMES, detrend_with_sinc_filter
from pykrait.trace_analysis.peak_analysis import (
    find_peaks,
    calculate_normalized_peaks,
    create_peak_properties,
)
from pykrait.trace_analysis.oscillations import (
    find_oscillating_rois,
    find_median_frequency,
)
from pykrait.trace_analysis.synchronicity import calculate_synchronicity_zscore
from pykrait.visualization.image_overlays import (
    create_synchronicity_image,
    create_heatmap_image,
)

logger = logging.getLogger(__name__)

# default index of the channel the intensity traces are extracted from. With the default
# motion correction settings this is the non-reference channel, i.e. the calcium indicator.
INTENSITY_CHANNEL = 0

# parameters that change the masks, and in addition the intensities extracted with them.
# Saved masks and intensities are only reused if these match the saved parameters.
# a provided frame interval within 0.5% of the one in the metadata is taken to be a rounded
# copy of it, and the more precise metadata value is used
FRAME_INTERVAL_MATCH_TOLERANCE = 0.005

SEGMENTATION_PARAMETERS = ("tproj_type", "CLAHE_normalize", "cellpose_model_path")
INTENSITY_PARAMETERS = ("intensity_channel",)


@dataclass
class AnalysisParameters:
    """This dataclass holds the modifiable parameters for calcium video analysis.

    :var tproj_type: Type of projection accross T-axis to be used for segmentation; 'std' computes standard deviation, 'sum' computes temporal sum, defaults to std.
    :vartype tproj_type: Literal['std', 'sum'], optional

    :var CLAHE_normalize: whether to apply CLAHE (Contrast Limited Adaptive Histogram Equalization) normalization to enhance image contrast.
    :vartype CLAHE_normalize: bool, optional

    :var cellpose_model_path: Path to the Cellpose model used for segmentation. Defaults to "cpsam" for the built-in CPSAM model.
    :vartype cellpose_model_path: str, optional

    :var frame_interval: time interval between consecutive video frames in seconds, defaults to None. If None provided, the frame interval obtained from the image metadata will be used or it will default to 1 second. The code will output a warning if a frame interval is provided which doesn't match with the metadata frame interval.
    :vartype frame_interval: float, optional

    :var sinc_filter_window: Window size for the sinc filter used to detrend the signal, specified in seconds. Converted to frame count internally.
    :vartype sinc_filter_window: float, optional

    :var peak_min_width: Minimum width at half-maximum of a detected peak in seconds. Converted to frames internally.
    :vartype peak_min_width: float, optional

    :var peak_max_width: Maximum width at half-maximum of a detected peak in seconds. Converted to frames internally.
    :vartype peak_max_width: float, optional

    :var peak_prominence: Minimum prominence of a detected peak, defined as the height difference between the peak and its surrounding baseline. Arbitrary units.
    :vartype peak_prominence: float, optional

    :var peak_min_height: Minimum height of a detected peak in arbitrary units.
    :vartype peak_min_height: float, optional

    :var periodicity_method: Method to determine periodicity of ROIs, either "cutoff" for fixed thresholds or "quantile" for quantile-based thresholds.
    :vartype periodicity_method: Literal['cutoff', 'quantile'], optional

    :var std_threshold: Threshold for the Standard Deviation, either quantile or cutoff depending on method.
    :vartype std_threshold: float, optional

    :var cov_threshold: Threshold for the CoV, either quantile or cutoff depending on method.
    :vartype cov_threshold: float, optional

    :var time_window: Time window in seconds within which peaks are considered synchronous. Calculated to frames internally. Defaults to 10 seconds.
    :vartype time_window: float, optional

    :var neighbour_tolerance: Maximum distance in micrometers used to detect neighboring cells. Converted to pixels internally.
    :vartype neighbour_tolerance: float, optional

    :var intensity_channel: Index of the channel of the original video the intensity traces are extracted from, defaults to 0. If the motion correction drops its reference channel, the index still refers to the original video.
    :vartype intensity_channel: int, optional

    :var motion_correction: Whether to correct the lateral (xy) motion of the video before segmentation and intensity extraction. Disabled by default.
    :vartype motion_correction: bool, optional

    :var motion_correction_method: Method used to estimate the motion; "global_shift" for a rigid translation, "warp_grid" for a mesh based optical flow and "warp_polynomial_fit" for a polynomial fit of the optical flow.
    :vartype motion_correction_method: Literal['global_shift', 'warp_grid', 'warp_polynomial_fit'], optional

    :var motion_correction_channel: Index of the channel the motion is estimated on, defaults to 1. Works best on a channel with stationary labeling. Single-channel videos fall back to channel 0.
    :vartype motion_correction_channel: int, optional

    :var motion_correction_template_frames: Number of frames at the beginning of the recording that are averaged into the alignment template.
    :vartype motion_correction_template_frames: int, optional

    :var motion_correction_average_frames: Number of frames averaged in a rolling window before the motion is estimated. The correction itself is always applied to the single frame.
    :vartype motion_correction_average_frames: int, optional

    :var motion_correction_template_filter: Factor to update the template with every corrected frame as ``template = (1 - factor) * template + factor * frame``. 0 keeps the template fixed.
    :vartype motion_correction_template_filter: float, optional

    :var motion_correction_grid_size: Number of mesh cells along each axis, only used for the "warp_grid" method.
    :vartype motion_correction_grid_size: int, optional

    :var motion_correction_polynomial_degree: Total degree of the fitted polynomial, only used for the "warp_polynomial_fit" method.
    :vartype motion_correction_polynomial_degree: int, optional

    :var motion_correction_mask_fov: Whether to set all pixels that left the field of view in any frame to 0 in the whole corrected video.
    :vartype motion_correction_mask_fov: bool, optional

    :var motion_correction_drop_reference_channel: Whether to leave the reference channel out of the saved motion-corrected video. It is only needed to estimate the motion and for the projection, so dropping it roughly halves the size of the saved video and the time needed to read it back. Note that a re-run that reuses the saved video can then no longer reproduce the projection over all channels.
    :vartype motion_correction_drop_reference_channel: bool, optional
    """

    tproj_type: Literal["std", "sum"] = "std"
    CLAHE_normalize: bool = True
    cellpose_model_path: str = "cpsam"
    frame_interval: float = None
    sinc_filter_window: float = 300
    peak_min_width: float = 1
    peak_max_width: float = 40
    peak_prominence: float = 1000
    peak_min_height: float = 80
    periodicity_method: Literal["cutoff", "quantile"] = "cutoff"
    std_threshold: float = 15
    cov_threshold: float = 0.2
    time_window: float = 10
    neighbour_tolerance: float = 10
    intensity_channel: int = 0
    motion_correction: bool = False
    motion_correction_method: Literal[
        "global_shift", "warp_grid", "warp_polynomial_fit"
    ] = "warp_grid"
    motion_correction_channel: int = 1
    motion_correction_template_frames: int = 10
    motion_correction_average_frames: int = 3
    motion_correction_template_filter: float = 0.0
    motion_correction_grid_size: int = 10
    motion_correction_polynomial_degree: int = 2
    motion_correction_mask_fov: bool = False
    motion_correction_drop_reference_channel: bool = False

    def to_pandas(self) -> pd.DataFrame:
        """Convert the dataclass to a pandas DataFrame."""
        return pd.DataFrame([asdict(self)])

    @classmethod
    def from_csv(cls, path: str) -> "AnalysisParameters":
        """
        Reads parameters saved with save_params. Entries that are not a parameter are
        ignored and parameters without a saved value keep their default, both with a warning.

        :param path: path to the analysis parameters CSV file
        :type path: str
        :return: returns the saved analysis parameters
        :rtype: AnalysisParameters
        """
        saved = _read_saved_csv(path)
        field_types = {f.name: f.type for f in fields(cls)}

        unknown_keys = sorted(set(saved) - set(field_types))
        if unknown_keys:
            warnings.warn(
                f"Saved parameters in {path} contain unknown entries {unknown_keys} "
                "(not part of the current AnalysisParameters); they will be ignored."
            )
        missing_keys = sorted(set(field_types) - set(saved))
        if missing_keys:
            warnings.warn(
                f"No saved value for parameters {missing_keys} in {path}; using defaults."
            )

        return cls(
            **{
                name: _cast_parameter(_normalize_parameter(value), field_types[name])
                for name, value in saved.items()
                if name in field_types
            }
        )


@dataclass
class AnalysisOutput:
    """
    This dataclass holds the output variables of the pipeline.
    :var filepath: Path to the calcium imaging video file.
    :vartype filepath: str
    :var filename: name of the video file
    :vartype filename: str
    :var tproj_path: Path to the zproj file.
    :vartype tproj_path: str
    :var masks_path: Path to the masks file.
    :vartype masks_path: str
    :var motion_corrected_path: Path to the motion-corrected video, or None if no motion correction was applied.
    :vartype motion_corrected_path: str
    :var analysis_folder: folder the analysis files are saved in
    :vartype analysis_folder: str
    :var frame_interval: time interval between frames in seconds
    :vartype frame_interval: float
    :var pixel_interval_y: pixel size in y direction in micrometers
    :vartype pixel_interval_y: float
    :var pixel_interval_x: pixel size in x direction in micrometers
    :vartype pixel_interval_x: float
    :var frame_interval_source: where the frame interval comes from: "metadata", "parameter" (given by the user) or "default" (1 second, as neither was available)
    :vartype frame_interval_source: str
    :var pixel_size_source: where the pixel size comes from: "metadata" or "default" (1 µm, as the metadata has none)
    :vartype pixel_size_source: str
    :var number_of_cells: number of cells detected in the video
    :vartype number_of_cells: int
    :var number_of_frames: number of frames in the video
    :vartype number_of_frames: int
    :var number_of_channels: number of channels in the original video
    :vartype number_of_channels: int
    :var normalized_peaks: number of peaks per 100 cells and 10 minutes of recording
    :vartype normalized_peaks: float
    :var cells_four_peaks: number of cells with at least 4 peaks detected
    :vartype cells_four_peaks: int
    :var std_cutoff: cutoff for STD threshold for the standard deviation in seconds
    :vartype std_cutoff: float
    :var std_quantile: quantile of random STDs compared to randomly shuffled peaks
    :vartype std_quantile: float
    :var random_below_std: number of cells with a standard deviation below the threshold in the random control
    :vartype random_below_std: int
    :var experimental_below_std: number of cells with a standard deviation below the threshold in the recording
    :vartype experimental_below_std: int
    :var cov_cutoff: cutoff for CoV (coefficient of variation) in the recording
    :vartype cov_cutoff: float
    :var cov_quantile: quantile of random CoV compared to randomly shuffled peaks
    :vartype cov_quantile: float
    :var random_below_cov: number of cells with a coefficient of variation below the threshold in the random control
    :vartype random_below_cov: int
    :var experimental_below_cov: number of cells with a coefficient of variation below the threshold in the recording
    :vartype experimental_below_cov: int
    :var median_frequency_mHz: median frequency of all peak-peak intervals in mHz
    :vartype median_frequency_mHz: float
    :var active_cells_median_frequency_mHz: median frequency of the median peak-peak intervals of cells with at least 4 peaks in mHz
    :vartype active_cells_median_frequency_mHz: float
    :var median_peakwidth_s: median peak width at half maximum in seconds
    :vartype median_peakwidth_s: float
    :var median_rise2080_s: median rise time from 20% to 80% of peak height in seconds
    :vartype median_rise2080_s: float
    :var median_decay8020_s: median decay time from 80% to 20% of peak height in seconds
    :vartype median_decay8020_s: float
    :var total_coinciding_peaks_1st_neighbour: total number of coinciding peaks between 1st degree neighbours
    :vartype total_coinciding_peaks_1st_neighbour: int
    :var possible_coinciding_peaks_1st_neighbour: total number of possible coinciding peaks between 1st degree neighbours
    :vartype possible_coinciding_peaks_1st_neighbour: int
    :var random_1st_neighbour_zscore: z-score of synchronicity of the first neighbours in the random control
    :vartype random_1st_neighbour_zscore: float
    :var random_2nd_neighbour_zscore: z-score of synchronicity of the second neighbours in the random control
    :vartype random_2nd_neighbour_zscore: float
    :var random_3rd_neighbour_zscore: z-score of synchronicity of the third neighbours in the random control
    :vartype random_3rd_neighbour_zscore: float
    :var experiment_1st_neighbour_zscore: z-score of synchronicity of the first neighbours in the recording
    :vartype experiment_1st_neighbour_zscore: float
    :var experiment_2nd_neighbour_zscore: z-score of synchronicity of the second neighbours in the recording
    :vartype experiment_2nd_neighbour_zscore: float
    :var experiment_3rd_neighbour_zscore: z-score of synchronicity of the third neighbours in the recording
    :vartype experiment_3rd_neighbour_zscore: float
    :var pykrait_version: version of the pykrait package used for the analysis
    :vartype pykrait_version: str
    :var timestamp: timestamp of the analysis
    :vartype timestamp: np.datetime64
    :var error: error message if the analysis failed, None otherwise
    :vartype error: str
    """

    filepath: str = None
    filename: str = None
    tproj_path: str = None
    masks_path: str = None
    motion_corrected_path: str = None
    analysis_folder: str = None
    frame_interval: float = None
    pixel_interval_y: float = None
    pixel_interval_x: float = None
    frame_interval_source: str = None
    pixel_size_source: str = None
    number_of_cells: int = None
    number_of_frames: int = None
    number_of_channels: int = None
    normalized_peaks: float = None
    cells_four_peaks: int = None
    std_cutoff: float = None
    std_quantile: float = None
    random_below_std: int = None
    experimental_below_std: int = None
    cov_cutoff: float = None
    cov_quantile: float = None
    random_below_cov: int = None
    experimental_below_cov: int = None
    median_frequency_mHz: float = None
    active_cells_median_frequency_mHz: float = None
    median_peakwidth_s: float = None
    median_rise2080_s: float = None
    median_decay8020_s: float = None
    total_coinciding_peaks_1st_neighbour: int = None
    possible_coinciding_peaks_1st_neighbour: int = None
    random_1st_neighbour_zscore: float = None
    random_2nd_neighbour_zscore: float = None
    random_3rd_neighbour_zscore: float = None
    experiment_1st_neighbour_zscore: float = None
    experiment_2nd_neighbour_zscore: float = None
    experiment_3rd_neighbour_zscore: float = None
    pykrait_version: str = None
    timestamp: np.datetime64 = None
    error: str = None

    def to_pandas(self):
        """Convert the dataclass to a pandas DataFrame."""
        return pd.DataFrame([asdict(self)])

    @classmethod
    def from_pandas(cls, df: pd.DataFrame) -> "AnalysisOutput":
        """Safely construct a dataclass from the first row of a pandas DataFrame."""
        if df.empty:
            raise ValueError("DataFrame is empty, cannot construct AnalysisOutput.")

        row = df.iloc[0].to_dict()
        init_args = {}

        for f in fields(cls):
            # outputs saved by older versions lack the fields added since
            if f.name not in row:
                init_args[f.name] = None
                continue

            val = row[f.name]

            # Handle missing / NaN
            if pd.isna(val):
                init_args[f.name] = None
                continue

            # Special case: np.datetime64
            if f.type == np.datetime64:
                init_args[f.name] = np.datetime64(val)
            else:
                try:
                    init_args[f.name] = f.type(val)  # cast to correct type
                except Exception:
                    raise TypeError(f"Cannot cast field '{f.name}' to {f.type}: {val}")

        return cls(**init_args)

    @classmethod
    def from_csv(cls, path: str) -> "AnalysisOutput":
        """Reads an analysis output saved with save_params."""
        return cls.from_pandas(pd.read_csv(path, float_precision="round_trip"))


class CalciumVideo:
    """
    Class to run the analysis pipeline on a single calcium video.
    """

    def __init__(self, video_path: str, params: AnalysisParameters):
        """
        initializes the calcium video analysis with the video path and the analysis parameters

        :param video_path: path to the calcium video
        :type video_path: str
        :param params: analysis parameters
        :type params: AnalysisParameters
        """
        self.analysis_parameters = params
        self.analysis_output = AnalysisOutput(filepath=video_path)

    def run(
        self,
        skip_cellpose: bool = False,
        skip_intensity_extraction: bool = False,
        raise_errors: bool = False,
        seed: int = None,
        video_cache: VideoCacheMode = "auto",
        save_temp_file: bool = True,
    ) -> AnalysisOutput:
        """
        runs the analysis pipeline on the calcium video and returns the analysis output

        Saved masks and intensities are only reused if they were created with the same
        segmentation, motion correction and intensity channel parameters, and intensities
        only together with the masks they were extracted from.

        :param skip_cellpose: whether to reuse existing masks instead of segmenting again, defaults to False
        :type skip_cellpose: bool, optional
        :param skip_intensity_extraction: whether to reuse existing intensities instead of extracting them again, defaults to False
        :type skip_intensity_extraction: bool, optional
        :param raise_errors: whether to raise errors instead of returning an empty AnalysisOutput with the error message, defaults to False
        :type raise_errors: bool, optional
        :param seed: if given, seeds numpy's global random state before the analysis of the traces, so that the random controls are reproducible, defaults to None
        :type seed: int, optional
        :param video_cache: where to keep the intensity channel between the first pass over the video (the motion correction, or else the projection) and the intensity extraction, so the video is only read once. "auto" keeps it in memory if it needs at most half of the available memory, otherwise in a temporary file if the video is on another drive than the temporary directory, see pykrait.io.frame_cache. "memory", "disk" or "none" force a location, defaults to "auto"
        :type video_cache: str, optional
        :param save_temp_file: whether the intensity channel may be kept in a temporary file on disk if it does not fit into memory. If False, such videos are read again for the intensities, defaults to True
        :type save_temp_file: bool, optional
        :return: returns the AnalysisOutput dataclass, containing the results of the analysis
        :rtype: AnalysisOutput
        """
        # every run starts from a clean output, so nothing carries over from a previous run
        self.analysis_output = AnalysisOutput(filepath=self.analysis_output.filepath)
        frame_cache = None
        try:
            # creating the analysis folder
            self.analysis_output = create_analysis_folder(self.analysis_output)

            filename_wo_ext = get_analysis_stem(self.analysis_output.filepath)
            analysis_folder = self.analysis_output.analysis_folder

            logger.info(f"Running analysis on {self.analysis_output.filename}")

            video_path = self.analysis_output.filepath
            tproj_path = get_analysis_file(video_path, "tproj.ome.tif", analysis_folder)
            cellpose_path = get_analysis_file(
                video_path, "cp_masks.ome.tif", analysis_folder
            )
            mean_intensities_path = get_analysis_file(
                video_path, "raw_intensities.csv.zst", analysis_folder
            )

            # loading the timelapse lazily and analysing frame interval and pixel size
            self.timelapse_data, self.analysis_output = load_timelapse(
                self.analysis_output, self.analysis_parameters
            )
            motion_corrected = False

            reuse_masks = (
                skip_cellpose
                and os.path.exists(tproj_path)
                and os.path.exists(cellpose_path)
                and not _saved_parameters_differ(
                    analysis_folder,
                    filename_wo_ext,
                    self.analysis_parameters,
                    SEGMENTATION_PARAMETERS,
                    "masks",
                )
            )
            if reuse_masks:
                logger.info(
                    f"Cellpose mask already exists for {self.analysis_output.filename}, loading existing results."
                )
                self.tproj = read_projection(tproj_path)
                self.masks = read_label_image(cellpose_path)
                self.rois = get_pixel_contour_from_label_img(self.masks)
                self.analysis_output.tproj_path = tproj_path
                self.analysis_output.masks_path = cellpose_path
            else:
                # the first pass over the video, the motion correction if it runs and the
                # projection otherwise, keeps the intensity channel, so the intensities
                # do not read the video again
                frame_cache = create_intensity_cache(
                    self.timelapse_data,
                    self.analysis_output,
                    self.analysis_parameters,
                    mode=video_cache,
                    save_temp_file=save_temp_file,
                )
                # correcting the lateral motion, if enabled in the parameters
                (
                    self.timelapse_data,
                    self.analysis_output,
                    raw_projection,
                ) = apply_motion_correction(
                    self.timelapse_data,
                    self.analysis_output,
                    self.analysis_parameters,
                    frame_cache=frame_cache,
                )
                motion_corrected = True
                # creating the masks and saving them to the analysis folder; without a
                # projection from the correction, reading the video fills the cache
                self.tproj, self.rois, self.masks, self.analysis_output = create_masks(
                    self.timelapse_data,
                    self.analysis_output,
                    self.analysis_parameters,
                    raw_projection=raw_projection,
                    frame_cache=frame_cache,
                )
                # the new masks are already saved, so the saved parameters have to
                # describe them now, and the intensities of the old masks are dropped.
                # Otherwise an interrupted run leaves masks that a later run reuses
                # under the parameters of the masks they replaced
                _remove_file(mean_intensities_path)
                self.analysis_parameters.to_pandas().to_csv(
                    get_analysis_file(
                        video_path, "analysis_parameters.csv", analysis_folder
                    ),
                    index=False,
                )

            require_rois(self.masks, self.analysis_output.filename)

            # intensities belong to the masks they were extracted from, so they can only
            # be reused together with those masks
            reuse_intensities = (
                skip_intensity_extraction
                and reuse_masks
                and os.path.exists(mean_intensities_path)
                and not _saved_parameters_differ(
                    analysis_folder,
                    filename_wo_ext,
                    self.analysis_parameters,
                    INTENSITY_PARAMETERS,
                    "intensities",
                )
            )
            if skip_intensity_extraction and not reuse_masks:
                logger.info(
                    "Masks were created again, so the intensities are extracted again as well."
                )
            if reuse_intensities:
                logger.info(
                    f"Mean intensities already exist for {self.analysis_output.filename}, loading existing results."
                )
                # intensities that do not match the masks, e.g. masks edited by hand,
                # raise instead of being assigned to the wrong ROIs
                self.mean_intensities = read_Txnrois(
                    mean_intensities_path, n_rois=int(self.masks.max())
                )
                (
                    self.analysis_output.number_of_frames,
                    self.analysis_output.number_of_cells,
                ) = self.mean_intensities.shape
                self.cell_properties = extract_cell_properties(self.masks)
            else:
                # the saved intensities no longer match, so an interrupted run cannot
                # leave them behind next to parameters that describe new ones
                _remove_file(mean_intensities_path)
                if not motion_corrected:
                    # a correction that runs keeps the intensity channel while correcting
                    if self.analysis_parameters.motion_correction:
                        frame_cache = create_intensity_cache(
                            self.timelapse_data,
                            self.analysis_output,
                            self.analysis_parameters,
                            mode=video_cache,
                            save_temp_file=save_temp_file,
                        )
                    # correcting the lateral motion, if enabled in the parameters
                    self.timelapse_data, self.analysis_output, raw_projection = (
                        apply_motion_correction(
                            self.timelapse_data,
                            self.analysis_output,
                            self.analysis_parameters,
                            frame_cache=frame_cache,
                        )
                    )
                    if raw_projection is None and frame_cache is not None:
                        # an existing corrected video was reused, so nothing filled the
                        # cache and the intensities read the video directly
                        frame_cache.close()
                        frame_cache = None
                # extracting the mean intensities from the timelapse data and saving them to the analysis folder
                self.mean_intensities, self.cell_properties, self.analysis_output = (
                    create_mean_intensities(
                        self.timelapse_data
                        if frame_cache is None
                        else frame_cache.as_timelapse(),
                        self.masks,
                        self.analysis_output,
                        channel_index=intensity_channel_index(
                            self.analysis_parameters,
                            self.analysis_output.number_of_channels,
                            self.timelapse_data.shape[1],
                        ),
                    )
                )

            # convert seconds and µm to frames and pixels
            neighbour_tolerance_pixels = neighbour_tolerance_in_pixels(
                self.analysis_parameters, self.analysis_output
            )
            peak_min_width_frames, peak_max_width_frames = peak_widths_in_frames(
                self.analysis_parameters, self.analysis_output
            )

            # the random controls of the periodicity and synchronicity draw from numpy's
            # global random state
            if seed is not None:
                np.random.seed(seed)

            self.detrended_timeseries = detrend_with_sinc_filter(
                signals=self.mean_intensities,
                cutoff_period=self.analysis_parameters.sinc_filter_window,
                sampling_interval=self.analysis_output.frame_interval,
            )

            (
                self.peak_series,
                peak_properties_df,
                peak_frame_nonzero,
                self.analysis_output,
            ) = do_peak_analysis(
                detrended_timeseries=self.detrended_timeseries,
                peak_min_width_frames=peak_min_width_frames,
                peak_max_width_frames=peak_max_width_frames,
                analysis_parameters=self.analysis_parameters,
                analysis_output=self.analysis_output,
            )

            self.analysis_output, self.experimental_stds, self.experimental_covs = (
                calculate_periodic_cells(
                    self.peak_series, self.analysis_parameters, self.analysis_output
                )
            )

            self.roi_dataframe = create_roi_dataframe(
                self.peak_series,
                self.experimental_stds,
                self.experimental_covs,
                self.analysis_output,
            )

            self.adjacency_matrix, self.shortest_path_matrix = get_adjacency_matrix(
                self.masks, neighbour_tolerance=neighbour_tolerance_pixels
            )

            (
                self.analysis_output,
                self.true_synchronous_peaks_matrix,
                self.rand_synchronous_peaks_matrix,
            ) = calculate_synchronicity(
                self.shortest_path_matrix,
                self.peak_series,
                self.analysis_parameters,
                self.analysis_output,
            )
            # saving the parameters and output to the analysis folder
            save_analysis_results(
                tproj=self.tproj,
                masks=self.masks,
                synchronous_peaks_matrix=self.true_synchronous_peaks_matrix,
                analysis_output=self.analysis_output,
                analysis_params=self.analysis_parameters,
                detrended_intensities=self.detrended_timeseries,
                peaks=self.peak_series,
                intensities=self.mean_intensities,
                stds=self.experimental_stds,
                covs=self.experimental_covs,
                roi_dataframe=self.roi_dataframe,
                overwrite=True,
            )
            return self.analysis_output

        except Exception as e:
            logger.error(
                f"Analysis failed for video {self.analysis_output.filename} with error: {e}\n{traceback.format_exc()}"
            )
            if raise_errors:
                raise
            return AnalysisOutput(
                filepath=self.analysis_output.filepath,
                filename=self.analysis_output.filename,
                error=f"{type(e).__name__}: {e}",
            )
        finally:
            if frame_cache is not None:
                frame_cache.close()


class BatchExperiment:
    """
    Class to run a batch experiment on multiple calcium videos in a folder.
    """

    def __init__(
        self,
        folder: str,
        params: AnalysisParameters,
        extension: str = ".czi",
        filter_string: str = None,
        exclude_string: str = None,
        recursive: bool = False,
        load_params: bool = False,
    ):
        """
        sets up a batch run for all videos in a folder with the given parameters. If load_params is True, it will try to load the parameters from a previous analysis in the folder for each video, otherwise it will use the provided parameters for all videos.

        Acquisitions that are not timelapses (snapshots, z-stacks, ...) usually live next to
        the recordings to be analyzed. Use `filter_string` and `exclude_string` to select only
        the videos of interest, and `recursive` if they are spread over subfolders.

        :param folder: folder that contains the calcium videos to be analyzed
        :type folder: str
        :param params: AnalysisParameters object containing the parameters for the analysis
        :type params: AnalysisParameters
        :param extension: File Extension of Calcium Videos, defaults to ".czi"
        :type extension: str, optional
        :param filter_string: if given, only videos whose file name matches this pattern are analyzed. Shell-style wildcards are supported ("*_timeseries.czi"), any other pattern is matched as a substring ("timeseries"), defaults to None
        :type filter_string: str, optional
        :param exclude_string: if given, videos whose file name matches this pattern are skipped, e.g. "_plane". Uses the same matching as `filter_string`, defaults to None
        :type exclude_string: str, optional
        :param recursive: whether to also analyze the videos in all subfolders, defaults to False
        :type recursive: bool, optional
        :param load_params: whether to load the parameters from a previous analysis in the folder, defaults to False
        :type load_params: bool, optional
        """
        files = get_files_from_folder(
            folder,
            extension=extension,
            filter_string=filter_string,
            exclude_string=exclude_string,
            recursive=recursive,
        )
        if not files:
            warnings.warn(
                f"No '{extension}' files found in {folder} "
                f"(filter_string={filter_string!r}, exclude_string={exclude_string!r}, recursive={recursive}). "
                "Nothing will be analyzed.",
                UserWarning,
            )
        else:
            logger.info(f"Found {len(files)} videos to analyze in {folder}")
        self.experiments = []

        for video_path in files:
            if load_params:
                param_path = get_analysis_file(video_path, "analysis_parameters.csv")
                if not os.path.exists(param_path):
                    warnings.warn(
                        f"No analysis parameters for video {os.path.basename(video_path)} found at {param_path}. Using the provided parameters."
                    )
                    video_params = replace(params)
                else:
                    video_params = AnalysisParameters.from_csv(param_path)
            else:
                # every video gets its own copy, so changing one does not change the others
                video_params = replace(params)

            self.experiments.append(
                CalciumVideo(video_path=video_path, params=video_params)
            )

        self.results = pd.DataFrame()

    def run(
        self,
        skip_cellpose: bool = False,
        skip_intensity_extraction: bool = False,
        raise_errors: bool = False,
        seed: int = None,
        video_cache: VideoCacheMode = "auto",
        save_temp_file: bool = True,
    ):
        """
        runs the analysis pipeline for each video in the folder and stores the results in a pandas DataFrame.
        Videos that fail are listed with their error message in the `error` column, unless `raise_errors` is set.

        :param skip_cellpose: whether to reuse existing masks instead of segmenting again, defaults to False
        :type skip_cellpose: bool, optional
        :param skip_intensity_extraction: whether to reuse existing intensities instead of extracting them again, defaults to False
        :type skip_intensity_extraction: bool, optional
        :param raise_errors: whether to stop at the first video that fails, defaults to False
        :type raise_errors: bool, optional
        :param seed: if given, the random controls of every video are seeded with it, see CalciumVideo.run, defaults to None
        :type seed: int, optional
        :param video_cache: where to keep the intensity channel between the two passes over each video, see CalciumVideo.run, defaults to "auto"
        :type video_cache: str, optional
        :param save_temp_file: whether the intensity channel may be kept in a temporary file on disk, see CalciumVideo.run, defaults to True
        :type save_temp_file: bool, optional
        """
        results = []
        for experiment in self.experiments:
            results.append(
                experiment.run(
                    skip_cellpose=skip_cellpose,
                    skip_intensity_extraction=skip_intensity_extraction,
                    raise_errors=raise_errors,
                    seed=seed,
                    video_cache=video_cache,
                    save_temp_file=save_temp_file,
                ).to_pandas()
            )
        self.results = (
            pd.concat(results, ignore_index=True) if results else pd.DataFrame()
        )

        failed = self.results["error"].notna().sum() if results else 0
        if failed:
            warnings.warn(
                f"Analysis failed for {failed} of {len(results)} videos, see the 'error' column of the results.",
                UserWarning,
            )


def create_analysis_folder(analysis_output: AnalysisOutput) -> AnalysisOutput:
    """
    creates an analysis subfolder in the parent directory of the video file

    :param analysis_output: AnalysisOutput object containing the filepath of the video file
    :type analysis_output: AnalysisOutput
    :return: returns the AnalysisOutput object with the analysis folder path set
    :rtype: AnalysisOutput
    """
    analysis_output.filename = os.path.basename(analysis_output.filepath)
    # Create the analysis folder next to the video file, or reuse the analysis folder the
    # video already lies in
    analysis_folder = get_analysis_folder(analysis_output.filepath)
    os.makedirs(analysis_folder, exist_ok=True)
    analysis_output.analysis_folder = analysis_folder
    return analysis_output


def load_timelapse(
    analysis_output: AnalysisOutput, analysis_parameters: AnalysisParameters
) -> Tuple[da.Array, AnalysisOutput]:
    """
    loads the timelapse lazily and sets the frame interval, pixel size and number of channels.
    A frame interval given in the parameters takes precedence over the one in the metadata,
    unless it lies within 0.5% of it. Then it is taken to be a rounded copy and the more
    precise metadata value is used.

    :param analysis_output: AnalysisOutput object containing the filepath of the video
    :type analysis_output: AnalysisOutput
    :param analysis_parameters: AnalysisParameters object containing the optional frame interval
    :type analysis_parameters: AnalysisParameters
    :raises ValueError: if the video has fewer than MIN_FRAMES frames
    :return: returns the timelapse as a Dask array of shape (T, C, Y, X) and the modified AnalysisOutput object
    :rtype: Tuple[da.Array, AnalysisOutput]
    """
    (
        timelapse_data,
        frame_interval,
        analysis_output.pixel_interval_y,
        analysis_output.pixel_interval_x,
    ) = load_timelapse_lazy(file_path=analysis_output.filepath)
    analysis_output.number_of_channels = timelapse_data.shape[1]
    if timelapse_data.shape[0] < MIN_FRAMES:
        raise ValueError(
            f"{os.path.basename(analysis_output.filepath)} has only {timelapse_data.shape[0]} frame(s), "
            f"but the analysis needs a timelapse of at least {MIN_FRAMES} frames. "
            "Is it a snapshot or a z-stack instead of a calcium video?"
        )

    analysis_output.pixel_size_source = "metadata"
    if (
        analysis_output.pixel_interval_x is None
        or analysis_output.pixel_interval_y is None
    ):
        analysis_output.pixel_size_source = "default"
        warnings.warn(
            "No pixel size found in metadata. Using default value of 1.0 µm, so neighbour_tolerance is interpreted in pixels.",
            UserWarning,
        )
        analysis_output.pixel_interval_y = analysis_output.pixel_interval_y or 1.0
        analysis_output.pixel_interval_x = analysis_output.pixel_interval_x or 1.0

    # a frame interval of 0 means "as fast as possible" in some acquisition software and
    # cannot be used to convert between frames and seconds
    if frame_interval is not None and frame_interval <= 0:
        warnings.warn(
            f"Ignoring the frame interval of {frame_interval} s found in the metadata.",
            UserWarning,
        )
        frame_interval = None

    # parameters read back from a CSV store a missing frame interval as NaN
    if analysis_parameters.frame_interval is not None and pd.isna(
        analysis_parameters.frame_interval
    ):
        analysis_parameters.frame_interval = None

    if (
        frame_interval is None and analysis_parameters.frame_interval is None
    ):  # nothing provided in metadata or parameters
        warnings.warn(
            "No frame interval found in metadata and no frame interval provided in parameters. Using default value of 1.0 seconds.",
            UserWarning,
        )
        analysis_output.frame_interval = 1.0
        analysis_output.frame_interval_source = "default"
    elif (
        frame_interval is None and analysis_parameters.frame_interval is not None
    ):  # no metadata, but parameter provided
        analysis_output.frame_interval = analysis_parameters.frame_interval
        analysis_output.frame_interval_source = "parameter"
    elif (
        frame_interval is not None and analysis_parameters.frame_interval is None
    ):  # metadata provided, but no parameter
        analysis_output.frame_interval = frame_interval
        analysis_output.frame_interval_source = "metadata"
    elif math.isclose(
        frame_interval,
        analysis_parameters.frame_interval,
        rel_tol=FRAME_INTERVAL_MATCH_TOLERANCE,
    ):  # both provided and match, the parameter is usually a rounded copy of the metadata
        analysis_output.frame_interval = frame_interval
        analysis_output.frame_interval_source = "metadata"
    else:
        warnings.warn(
            f"Provided frame_interval ({analysis_parameters.frame_interval}) does not match inferred frame_interval ({frame_interval}) from metadata. Using provided value."
        )
        analysis_output.frame_interval = analysis_parameters.frame_interval
        analysis_output.frame_interval_source = "parameter"
    return timelapse_data, analysis_output


def neighbour_tolerance_in_pixels(
    analysis_parameters: AnalysisParameters, analysis_output: AnalysisOutput
) -> int:
    """
    Converts the neighbour tolerance from micrometers to pixels, at least 1 pixel.

    :param analysis_parameters: AnalysisParameters object containing the neighbour tolerance in µm
    :type analysis_parameters: AnalysisParameters
    :param analysis_output: AnalysisOutput object containing the pixel size
    :type analysis_output: AnalysisOutput
    :return: returns the neighbour tolerance in pixels
    :rtype: int
    """
    neighbour_tolerance_pixels = int(
        analysis_parameters.neighbour_tolerance / analysis_output.pixel_interval_x
    )
    if neighbour_tolerance_pixels < 1:
        warnings.warn(
            f"Neighbour tolerance of {neighbour_tolerance_pixels} pixels is too small, setting to 1 pixel."
        )
        neighbour_tolerance_pixels = 1
    return neighbour_tolerance_pixels


def peak_widths_in_frames(
    analysis_parameters: AnalysisParameters, analysis_output: AnalysisOutput
) -> Tuple[int, int]:
    """
    Converts the minimum and maximum peak width from seconds to frames. The minimum width
    is at least 1 frame and the maximum width larger than the minimum width.

    :param analysis_parameters: AnalysisParameters object containing the peak widths in seconds
    :type analysis_parameters: AnalysisParameters
    :param analysis_output: AnalysisOutput object containing the frame interval
    :type analysis_output: AnalysisOutput
    :return: returns the minimum and maximum peak width in frames
    :rtype: Tuple[int, int]
    """
    peak_min_width_frames = int(
        analysis_parameters.peak_min_width / analysis_output.frame_interval
    )
    peak_max_width_frames = int(
        analysis_parameters.peak_max_width / analysis_output.frame_interval
    )
    if peak_min_width_frames < 1:
        warnings.warn(
            f"Peak minimum width of {peak_min_width_frames} frames is too small, setting to 1 frame."
        )
        peak_min_width_frames = 1
    if peak_max_width_frames <= peak_min_width_frames:
        warnings.warn(
            f"Peak maximum width of {peak_max_width_frames} frames is smaller or equal than peak minimum width of {peak_min_width_frames} seconds, setting to peak minimum width + 1."
        )
        peak_max_width_frames = peak_min_width_frames + 1
    return peak_min_width_frames, peak_max_width_frames


def intensity_channel_index(
    analysis_parameters: AnalysisParameters, n_original_channels: int, n_channels: int
) -> int:
    """
    Maps the intensity channel, given as index into the original video, onto a video or
    projection with `n_channels` channels. That is either the original video, or the
    motion-corrected one with the reference channel dropped.

    :param analysis_parameters: AnalysisParameters object containing the intensity channel and motion correction parameters
    :type analysis_parameters: AnalysisParameters
    :param n_original_channels: number of channels of the original video, None if unknown
    :type n_original_channels: int
    :param n_channels: number of channels of the video or projection to index
    :type n_channels: int
    :raises ValueError: if the intensity channel does not exist or is the dropped reference channel
    :return: returns the index of the intensity channel in the video or projection
    :rtype: int
    """
    channel = int(analysis_parameters.intensity_channel)
    if n_original_channels is None:
        n_original_channels = n_channels
    if not 0 <= channel < n_original_channels:
        raise ValueError(
            f"intensity_channel {channel} does not exist in a video with {n_original_channels} channels."
        )
    if n_channels == n_original_channels:
        return channel

    # the reference channel was dropped from the motion-corrected video
    reference_channel = _resolve_reference_channel(
        analysis_parameters.motion_correction_channel, n_original_channels
    )
    if channel == reference_channel:
        raise ValueError(
            f"intensity_channel {channel} is the motion correction reference channel, which was dropped from the motion-corrected video."
        )
    return channel - 1 if channel > reference_channel else channel


def require_rois(masks: np.ndarray, filename: str = None) -> None:
    """
    Checks that a label image contains at least one ROI, so that a segmentation that
    found no cells stops with a clear message instead of failing later in the analysis.

    :param masks: label image of the ROIs
    :type masks: np.ndarray
    :param filename: name of the video for the message, defaults to None
    :type filename: str, optional
    :raises ValueError: if the label image contains no ROIs
    """
    if masks is None or not np.any(masks):
        where = f" in {filename}" if filename else ""
        raise ValueError(
            f"No cells were found{where}: the masks contain no ROIs. Check the Cellpose "
            "model and the projection type, or the label image."
        )


def _remove_file(path: str) -> None:
    """Deletes a file if it exists."""
    if os.path.exists(path):
        release_tiff_handles(path)
        os.remove(path)


def _replace_file(source: str, destination: str) -> None:
    """Moves a file to `destination`, replacing the file there."""
    # Windows cannot replace or move a file that a lazily read array keeps open
    release_tiff_handles(source)
    release_tiff_handles(destination)
    os.replace(source, destination)


def _normalize_parameter(value):
    """Converts a parameter read back from a CSV into a plain Python value, NaN into None."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _cast_parameter(value, field_type):
    """Casts a number read back from a CSV to the type of its field, e.g. 300 to 300.0."""
    if value is None or isinstance(value, bool):
        return value
    if field_type is float and isinstance(value, int):
        return float(value)
    if field_type is int and isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _read_saved_csv(path: str) -> dict:
    """Reads the first row of a saved CSV file, with floats exactly as they were written."""
    return pd.read_csv(path, float_precision="round_trip").iloc[0].to_dict()


def _saved_parameters_differ(
    analysis_folder: str,
    stem: str,
    analysis_parameters: AnalysisParameters,
    names: Tuple[str, ...],
    what: str,
) -> bool:
    """
    Checks whether saved results were created with different parameters, by comparing the
    given parameters and all motion correction parameters with the parameters saved in
    the analysis folder. Parameters missing from the saved file, e.g. because they were
    added in a later version, are compared with their default value.

    :param analysis_folder: analysis folder of the video
    :type analysis_folder: str
    :param stem: file name stem of the analysis files
    :type stem: str
    :param analysis_parameters: current analysis parameters
    :type analysis_parameters: AnalysisParameters
    :param names: names of the parameters the saved results depend on
    :type names: Tuple[str, ...]
    :param what: description of the saved results for the messages
    :type what: str
    :return: returns True if any of the parameters changed
    :rtype: bool
    """
    param_path = os.path.join(analysis_folder, f"{stem}_analysis_parameters.csv")
    if not os.path.exists(param_path):
        warnings.warn(
            f"No saved parameters found at {param_path}, so it cannot be checked whether the saved {what} match the current parameters. Reusing them.",
            UserWarning,
        )
        return False

    saved = _read_saved_csv(param_path)
    defaults = asdict(AnalysisParameters())
    current = asdict(analysis_parameters)
    motion_correction_names = [
        name for name in current if name.startswith("motion_correction")
    ]
    saved_motion_correction = _normalize_parameter(
        saved.get("motion_correction", defaults["motion_correction"])
    )
    if not current["motion_correction"] and not saved_motion_correction:
        # the settings of a disabled motion correction do not affect anything
        motion_correction_names = ["motion_correction"]
    names = list(names) + motion_correction_names

    changed = [
        name
        for name in names
        if _normalize_parameter(saved.get(name, defaults[name]))
        != _normalize_parameter(current[name])
    ]
    if changed:
        logger.info(
            f"Parameters {changed} changed since the saved {what} were created, creating them again."
        )
        return True
    return False


def _motion_correction_settings(analysis_parameters: AnalysisParameters) -> dict:
    """
    Collects all motion correction fields of the analysis parameters. They are stored
    next to a motion-corrected video to detect whether it can be reused.

    :param analysis_parameters: AnalysisParameters object containing the motion correction parameters
    :type analysis_parameters: AnalysisParameters
    :return: returns the motion correction settings as a JSON-compatible dictionary
    :rtype: dict
    """
    settings = {
        field.name: getattr(analysis_parameters, field.name)
        for field in fields(analysis_parameters)
        if field.name.startswith("motion_correction")
    }
    # normalize numpy scalars that come from parameters read back from a CSV
    return json.loads(json.dumps(settings, default=str))


# The motion-corrected video is written as an uncompressed ImageJ hyperstack. Its frames lie
# back to back and the first page describes all of them, so a reader finds every frame
# without walking the headers of the other pages, which costs one seek per page on a hard
# drive. Lossless compression only saves a few percent on noisy calcium videos.
MOTION_CORRECTED_SUFFIX = "motion_corrected.tif"
# OME-TIFF written by earlier versions, still reused if it was created with the same settings
LEGACY_MOTION_CORRECTED_SUFFIX = "motion_corrected.ome.tif"
# classic TIFF stores the offsets of the pages in 32 bits, so larger ImageJ files are written
# with the first page only, which ImageJ and tifffile read through its image description
_IMAGEJ_MAX_FULL_SIZE = 2**32 - 2**25
# pixel types an ImageJ hyperstack can hold
_IMAGEJ_DTYPES = (np.uint8, np.uint16, np.float32)


def corrected_video_shape(
    shape: Tuple[int, int, int, int], analysis_parameters: AnalysisParameters
) -> Tuple[int, int, int, int]:
    """
    Returns the (T, C, Y, X) shape of the video the analysis reads, which is the
    motion-corrected one if the motion correction is enabled.

    :param shape: (T, C, Y, X) shape of the original video
    :type shape: Tuple[int, int, int, int]
    :param analysis_parameters: AnalysisParameters object containing the motion correction parameters
    :type analysis_parameters: AnalysisParameters
    :return: returns the (T, C, Y, X) shape of the analysed video
    :rtype: Tuple[int, int, int, int]
    """
    n_timepoints, n_channels, height, width = shape
    if (
        analysis_parameters.motion_correction
        and analysis_parameters.motion_correction_drop_reference_channel
        and n_channels > 1
    ):
        n_channels -= 1
    return (n_timepoints, n_channels, height, width)


def create_intensity_cache(
    timelapse_data: da.Array,
    analysis_output: AnalysisOutput,
    analysis_parameters: AnalysisParameters,
    mode: VideoCacheMode = "auto",
    save_temp_file: bool = True,
) -> FrameCache:
    """
    Creates a cache for the intensity channel of the analysed video, so that the intensity
    extraction does not read the video file again. It is filled by the first pass over the
    video: the motion correction if it runs, otherwise the projection.

    :param timelapse_data: lazily loaded original timelapse of shape (T, C, Y, X)
    :type timelapse_data: da.Array
    :param analysis_output: AnalysisOutput object containing the path and number of channels of the original video
    :type analysis_output: AnalysisOutput
    :param analysis_parameters: AnalysisParameters object containing the intensity channel and motion correction parameters
    :type analysis_parameters: AnalysisParameters
    :param mode: where to keep the frames, see pykrait.io.frame_cache.choose_video_cache, defaults to "auto"
    :type mode: VideoCacheMode, optional
    :param save_temp_file: whether the frames may be kept in a temporary file on disk, defaults to True
    :type save_temp_file: bool, optional
    :return: returns the cache, or None if the frames are not kept
    :rtype: FrameCache
    """
    shape = corrected_video_shape(timelapse_data.shape, analysis_parameters)
    channel = intensity_channel_index(
        analysis_parameters, analysis_output.number_of_channels, shape[1]
    )
    return FrameCache.for_shape(
        shape,
        timelapse_data.dtype,
        channel,
        video_path=analysis_output.filepath,
        mode=mode,
        save_temp_file=save_temp_file,
    )


def apply_motion_correction(
    timelapse_data: da.Array,
    analysis_output: AnalysisOutput,
    analysis_parameters: AnalysisParameters,
    overwrite: bool = False,
    verbose: bool = True,
    frame_cache: FrameCache = None,
) -> Tuple[da.Array, AnalysisOutput, np.ndarray]:
    """
    Corrects the lateral motion of a timelapse and saves the result to the analysis folder.

    If motion correction is disabled in the analysis parameters, the timelapse is returned
    unchanged. If a motion-corrected file that was created with the same settings already
    exists for this video, it is loaded instead of being computed again, unless
    `overwrite` is set. Motion-corrected OME-TIFF files written by earlier versions are
    reused as well.

    The corrected video is saved as an uncompressed ImageJ TIFF, which can be opened
    without reading the header of every frame, see MOTION_CORRECTED_SUFFIX.

    :param timelapse_data: lazily loaded timelapse data as a Dask array of shape (T, C, Y, X)
    :type timelapse_data: da.Array
    :param analysis_output: AnalysisOutput object to be modified with the path to the motion-corrected video
    :type analysis_output: AnalysisOutput
    :param analysis_parameters: AnalysisParameters object containing the motion correction parameters
    :type analysis_parameters: AnalysisParameters
    :param overwrite: whether to recompute an already existing motion-corrected video, defaults to False
    :type overwrite: bool, optional
    :param verbose: if True, shows a progress bar for the correction, defaults to True
    :type verbose: bool, optional
    :param frame_cache: cache of one channel of the corrected video, see create_intensity_cache. It is filled while correcting, so the intensities do not read the corrected video again. If the correction does not run, it is left empty, defaults to None
    :type frame_cache: FrameCache, optional
    :raises ValueError: if no analysis folder has been created yet
    :return: returns the (corrected) timelapse data, the modified AnalysisOutput object and the raw temporal projection accumulated while correcting (None if the correction did not run, then `frame_cache` is not filled either)
    :rtype: Tuple[da.Array, AnalysisOutput, np.ndarray]
    """
    if not analysis_parameters.motion_correction:
        return timelapse_data, analysis_output, None

    if analysis_output.analysis_folder is None:
        raise ValueError(
            "No analysis folder found. Run create_analysis_folder before the motion correction."
        )

    corrected_path = get_analysis_file(
        analysis_output.filepath,
        MOTION_CORRECTED_SUFFIX,
        analysis_output.analysis_folder,
    )
    legacy_path = get_analysis_file(
        analysis_output.filepath,
        LEGACY_MOTION_CORRECTED_SUFFIX,
        analysis_output.analysis_folder,
    )
    settings_path = get_analysis_file(
        analysis_output.filepath,
        "motion_correction.json",
        analysis_output.analysis_folder,
    )
    settings = _motion_correction_settings(analysis_parameters)

    # an existing file is only reused if it was created with the same settings
    existing_path = next(
        (path for path in (corrected_path, legacy_path) if os.path.exists(path)), None
    )
    reusable = existing_path is not None and not overwrite
    if reusable and os.path.exists(settings_path):
        with open(settings_path, "r") as settings_file:
            if json.load(settings_file) != settings:
                logger.info(
                    "Motion correction settings changed, correcting the video again..."
                )
                reusable = False
    elif reusable:
        warnings.warn(
            f"No motion correction settings found next to {existing_path}. Reusing the existing file, delete it to correct the video again.",
            UserWarning,
        )

    raw_projection = None
    if reusable:
        corrected_path = existing_path
        logger.info(
            f"Motion-corrected video already exists for {analysis_output.filename}, loading existing results."
        )
        if existing_path == legacy_path:
            logger.info(
                "It was saved as OME-TIFF by an earlier version, which opens slowly from hard drives. Delete it to save it again as ImageJ TIFF."
            )
    else:
        logger.info(
            f"Correcting lateral motion with method '{analysis_parameters.motion_correction_method}'..."
        )
        # the video is written under a temporary name, so an interrupted correction is
        # never mistaken for a finished one. An existing video keeps its settings until
        # the new one replaces it, so after an interruption it is still recognised as
        # corrected with other settings instead of being reused without settings
        partial_path = corrected_path + ".partial.tif"
        try:
            raw_projection = _correct_and_save(
                timelapse_data,
                partial_path,
                analysis_parameters,
                analysis_output=analysis_output,
                verbose=verbose,
                frame_cache=frame_cache,
            )
        except BaseException:
            _remove_file(partial_path)
            raise
        _remove_file(settings_path)
        _replace_file(partial_path, corrected_path)
        # the old video was corrected with other settings and replaced by this one
        _remove_file(legacy_path)
        with open(settings_path, "w") as settings_file:
            json.dump(settings, settings_file, indent=2)
        logger.info(f"Motion-corrected video saved in: {corrected_path}")

    analysis_output.motion_corrected_path = corrected_path
    corrected_data, _, _, _ = load_timelapse_lazy(file_path=corrected_path)
    return corrected_data, analysis_output, raw_projection


def _write_corrected_video(
    path: str,
    pages,
    shape: Tuple[int, int, int, int],
    dtype,
    analysis_output: AnalysisOutput = None,
) -> None:
    """
    Writes a (T, C, Y, X) video page by page as an uncompressed ImageJ hyperstack, with
    the frame interval and pixel size of the original video. Pixel types ImageJ does not
    support are written as uncompressed OME-TIFF instead.

    Default values the analysis falls back to, because the metadata has none, are not
    written, so the corrected video is just as uncalibrated as the original. A frame
    interval given in the parameters is written.

    :param path: path of the TIFF file to write
    :type path: str
    :param pages: iterable of the (Y, X) pages in TCYX order
    :param shape: (T, C, Y, X) shape of the video
    :type shape: Tuple[int, int, int, int]
    :param dtype: pixel type of the video
    :param analysis_output: AnalysisOutput object containing the frame interval and pixel size, defaults to None
    :type analysis_output: AnalysisOutput, optional
    """
    dtype = np.dtype(dtype)
    if dtype.type not in _IMAGEJ_DTYPES:
        with tifffile.TiffWriter(path, ome=True) as writer:
            writer.write(pages, shape=shape, dtype=dtype, metadata={"axes": "TCYX"})
        return

    metadata = {"axes": "TCYX"}
    resolution = None
    if analysis_output is not None:
        if (
            analysis_output.frame_interval
            and analysis_output.frame_interval_source != "default"
        ):
            metadata["finterval"] = float(analysis_output.frame_interval)
        if (
            analysis_output.pixel_interval_x
            and analysis_output.pixel_interval_y
            and analysis_output.pixel_size_source != "default"
        ):
            resolution = (
                1 / analysis_output.pixel_interval_x,
                1 / analysis_output.pixel_interval_y,
            )
            metadata["unit"] = "um"
    n_bytes = int(np.prod(shape)) * dtype.itemsize
    with tifffile.TiffWriter(path, imagej=True) as writer:
        writer.write(
            pages,
            shape=shape,
            dtype=dtype,
            metadata=metadata,
            resolution=resolution,
            truncate=n_bytes > _IMAGEJ_MAX_FULL_SIZE,
        )


def _correct_and_save(
    timelapse_data: da.Array,
    corrected_path: str,
    analysis_parameters: AnalysisParameters,
    analysis_output: AnalysisOutput = None,
    verbose: bool = True,
    frame_cache: FrameCache = None,
) -> np.ndarray:
    """
    Corrects the lateral motion of a timelapse and writes it out frame by frame, while
    accumulating the temporal projection of the corrected frames and keeping the
    intensity channel on the fly.

    Streaming the correction straight into the file keeps the memory use independent of
    the length of the recording, and accumulating the projection and keeping the intensity
    channel here means the corrected video does not have to be read again in this run.

    :param timelapse_data: lazily loaded timelapse data as a Dask array of shape (T, C, Y, X)
    :type timelapse_data: da.Array
    :param corrected_path: path the motion-corrected video is written to
    :type corrected_path: str
    :param analysis_parameters: AnalysisParameters object containing the motion correction parameters
    :type analysis_parameters: AnalysisParameters
    :param analysis_output: AnalysisOutput object whose frame interval and pixel size are saved with the video, defaults to None
    :type analysis_output: AnalysisOutput, optional
    :param verbose: if True, shows a progress bar for the correction, defaults to True
    :type verbose: bool, optional
    :param frame_cache: cache of one channel of the corrected video, filled while correcting, defaults to None
    :type frame_cache: FrameCache, optional
    :raises ValueError: if the frame cache does not match the shape of the corrected video
    :return: returns the raw temporal projection of the corrected video
    :rtype: np.ndarray
    """
    corrector = LateralMotionCorrector(
        timelapse_data,
        method=analysis_parameters.motion_correction_method,
        reference_channel=analysis_parameters.motion_correction_channel,
        template_frames=analysis_parameters.motion_correction_template_frames,
        average_frames=analysis_parameters.motion_correction_average_frames,
        template_filter=analysis_parameters.motion_correction_template_filter,
        grid_size=analysis_parameters.motion_correction_grid_size,
        polynomial_degree=analysis_parameters.motion_correction_polynomial_degree,
    )

    n_timepoints, n_channels, height, width = timelapse_data.shape
    # the reference channel is only needed to estimate the motion and for the projection,
    # so it can be left out of the saved video to halve its size
    if analysis_parameters.motion_correction_drop_reference_channel and n_channels > 1:
        kept_channels = [
            channel
            for channel in range(n_channels)
            if channel != corrector.reference_channel
        ]
        logger.info(
            f"Dropping reference channel {corrector.reference_channel} from the motion-corrected video"
        )
    else:
        kept_channels = list(range(n_channels))
    corrected_shape = (n_timepoints, len(kept_channels), height, width)
    if frame_cache is not None and frame_cache.shape != corrected_shape:
        raise ValueError(
            f"The frame cache has shape {frame_cache.shape}, but the corrected video has shape {corrected_shape}."
        )

    projection = TemporalProjection(method=analysis_parameters.tproj_type)

    def pages():
        """Yields the corrected frames page by page, one page per channel."""
        for timepoint, frame in enumerate(corrector.iter_frames(verbose=verbose)):
            projection.update(frame)  # the projection always uses every channel
            kept = frame[kept_channels]
            if frame_cache is not None:
                frame_cache.store(timepoint, kept[np.newaxis])
            yield from kept

    _write_corrected_video(
        corrected_path,
        pages(),
        corrected_shape,
        timelapse_data.dtype,
        analysis_output=analysis_output,
    )

    if analysis_parameters.motion_correction_mask_fov:
        # the valid field of view is only known once every frame has been corrected, so
        # the video is masked in a second pass
        _mask_field_of_view(
            corrected_path,
            corrector.valid_fov,
            analysis_output=analysis_output,
            verbose=verbose,
        )
        if frame_cache is not None:
            for start in range(0, n_timepoints, FRAME_CACHE_MASK_BLOCK):
                block = frame_cache.frames[start : start + FRAME_CACHE_MASK_BLOCK]
                block *= corrector.valid_fov.astype(block.dtype)[np.newaxis]

    # pixels that left the field of view are 0 in some frames, which makes them stand out
    # in the projection, so they are left out of the image that is segmented. A pixel that
    # is 0 in every frame has a std and sum projection of 0, so this is also the
    # projection of the masked video.
    return projection.result() * corrector.valid_fov[None, :, :]


# frames of the frame cache masked at once, so a cache on disk is never loaded whole
FRAME_CACHE_MASK_BLOCK = 64


def _mask_field_of_view(
    corrected_path: str,
    valid_fov: np.ndarray,
    analysis_output: AnalysisOutput = None,
    verbose: bool = True,
) -> None:
    """
    Sets all pixels that left the field of view in any frame to 0, rewriting the video
    block by block so that it never has to be held in memory as a whole.

    :param corrected_path: path of the motion-corrected video
    :type corrected_path: str
    :param valid_fov: binary mask that is 0 where a pixel left the field of view
    :type valid_fov: np.ndarray
    :param analysis_output: AnalysisOutput object whose frame interval and pixel size are saved with the video, defaults to None
    :type analysis_output: AnalysisOutput, optional
    :param verbose: if True, shows a progress bar, defaults to True
    :type verbose: bool, optional
    """
    masked_path = corrected_path + ".masking.tif"
    corrected_data, _, _, _ = load_timelapse_lazy(file_path=corrected_path)
    n_timepoints = corrected_data.shape[0]
    block_size = frames_per_block(corrected_data)

    def pages():
        with tqdm(
            total=n_timepoints, desc="Masking field of view", disable=not verbose
        ) as progress:
            for start in range(0, n_timepoints, block_size):
                block = np.asarray(corrected_data[start : start + block_size])
                block = block * valid_fov[None, None, :, :]
                for frame in block:
                    for channel_frame in frame:
                        yield channel_frame
                progress.update(block.shape[0])

    _write_corrected_video(
        masked_path,
        pages(),
        corrected_data.shape,
        corrected_data.dtype,
        analysis_output=analysis_output,
    )
    _replace_file(masked_path, corrected_path)


def create_masks(
    timelapse_data: da.Array,
    analysis_output: AnalysisOutput,
    analysis_parameters: AnalysisParameters,
    raw_projection: np.ndarray = None,
    frame_cache: FrameCache = None,
) -> Tuple[np.ndarray, list, np.ndarray, AnalysisOutput]:
    """
    creates a cellpose segmentation mask from the timelapse data and saves it to the analysis folder.

    Returns the tproj, the masks and the modified AnalysisOutput object with the mask path set.

    :param timelapse_data: lazily loaded timelapse data as a Dask array of shape (T, C, Y, X)
    :type timelapse_data: da.Array
    :param analysis_output: AnalysisOutput object to be modified with the mask path and other analysis results
    :type analysis_output: AnalysisOutput
    :param analysis_parameters: AnalysisParameters object containing the parameters for the analysis
    :type analysis_parameters: AnalysisParameters
    :param raw_projection: raw temporal projection that was already accumulated elsewhere, e.g. while correcting the motion. If given, the timelapse does not have to be read again, defaults to None
    :type raw_projection: np.ndarray, optional
    :param frame_cache: if given, keeps the frames of one channel while the timelapse is read for the projection, defaults to None
    :type frame_cache: FrameCache, optional
    :return: returns the projection, the ROI contours, the masks and the modified AnalysisOutput object with the mask path set
    :rtype: Tuple[np.ndarray, list, np.ndarray, AnalysisOutput]
    """
    # perform the tproj
    if raw_projection is not None:
        logger.info(
            "Using the T Projection accumulated during the motion correction..."
        )
        tproj = finalize_projection(
            raw_projection, normalize=analysis_parameters.CLAHE_normalize
        )
    else:
        logger.info("Performing T Projection...")
        tproj = timelapse_projection(
            timelapse_data,
            method=analysis_parameters.tproj_type,
            normalize=analysis_parameters.CLAHE_normalize,
            verbose=True,
            frame_cache=frame_cache,
        )

    analysis_output.tproj_path = get_analysis_file(
        analysis_output.filename, "tproj.ome.tif", analysis_output.analysis_folder
    )
    tifffile.imwrite(
        analysis_output.tproj_path,
        tproj.astype(np.uint16),
        metadata={"axes": "CYX"},
        compression="zlib",
    )

    # create cellpose segmentation
    logger.info("Performing Cellpose Segmentation...")
    masks = create_cellpose_segmentation(
        tproj, cellpose_model_path=analysis_parameters.cellpose_model_path
    )
    # the same cleanup as for loaded masks, so fresh and reused runs give the same ROIs
    masks = clean_label_image(masks)
    masks_path = get_analysis_file(
        analysis_output.filename, "cp_masks.ome.tif", analysis_output.analysis_folder
    )
    analysis_output.masks_path = masks_path
    tifffile.imwrite(
        masks_path, masks.astype(np.uint16), metadata={"axes": "YX"}, compression="zlib"
    )

    rois = get_pixel_contour_from_label_img(masks)

    return tproj, rois, masks, analysis_output


def create_mean_intensities(
    timelapse_data: da.Array,
    masks: np.ndarray,
    analysis_output: AnalysisOutput,
    channel_index: int = INTENSITY_CHANNEL,
) -> Tuple[np.ndarray, tuple, AnalysisOutput]:
    """
    wrapper to extract mean intensities from the timelapse data and update the analysis output

    :param timelapse_data: lazily loaded timelapse data as a Dask array
    :type timelapse_data: da.Array
    :param masks: label image of the ROIs
    :type masks: np.ndarray
    :param analysis_output: object to be modified with the number of frames and cells
    :type analysis_output: AnalysisOutput
    :param channel_index: index of the channel in `timelapse_data` to extract the intensities from, defaults to INTENSITY_CHANNEL
    :type channel_index: int, optional
    :return: returns the T x n_roi mean intensities, the cell properties and the modified AnalysisOutput object
    :rtype: Tuple[np.ndarray, tuple, AnalysisOutput]
    """
    logger.info("Extracting Mean Intensities...")
    mean_intensities = extract_mean_intensities(
        timelapse_data, masks, verbose=True, channel_index=channel_index
    )
    analysis_output.number_of_frames, analysis_output.number_of_cells = (
        mean_intensities.shape
    )
    cell_properties = extract_cell_properties(masks)
    return mean_intensities, cell_properties, analysis_output


def do_peak_analysis(
    detrended_timeseries: np.ndarray,
    peak_min_width_frames: int,
    peak_max_width_frames: int,
    analysis_parameters: AnalysisParameters,
    analysis_output: AnalysisOutput,
) -> Tuple[np.ndarray, pd.DataFrame, np.ndarray, AnalysisOutput]:
    """
    Wrapper function for peak analysis. It detects peaks, computes the normalized_peaks, analyses the peak properties and frequency.

    :param detrended_timeseries: T x n_roi timeseries with already detrended signals
    :type detrended_timeseries: np.ndarray
    :param peak_min_width_frames: minimum peak width in frames
    :type peak_min_width_frames: int
    :param peak_max_width_frames: maximum peak width in frames
    :type peak_max_width_frames: int
    :param analysis_parameters: analysis parameter object
    :type analysis_parameters: AnalysisParameters
    :param analysis_output: analysis output object
    :type analysis_output: AnalysisOutput
    :return: returns the peak series (binary T x n_roi where 1 = peak), the peak properties dataframe, an array with the 70 frames around the peak for visualization, and the modified analysis output object
    :rtype: Tuple[np.ndarray, pd.DataFrame, np.ndarray, AnalysisOutput]
    """
    # find peaks
    peak_series = find_peaks(
        peak_min_width=peak_min_width_frames,
        peak_max_width=peak_max_width_frames,
        peak_prominence=analysis_parameters.peak_prominence,
        peak_min_height=analysis_parameters.peak_min_height,
        detrended_timeseries=detrended_timeseries,
    )

    # compute normalized peaks
    peaks_per_100c_per_10_min = calculate_normalized_peaks(
        peak_series, analysis_output.frame_interval
    )
    analysis_output.normalized_peaks = round(peaks_per_100c_per_10_min, 2)

    # compute number of cells with at least 4 peaks
    analysis_output.cells_four_peaks = np.sum(np.sum(peak_series, axis=0) >= 4)

    # compute peak properties
    peak_properties_df, peak_frame_nonzero = create_peak_properties(
        peak_series, detrended_timeseries, analysis_output.frame_interval
    )
    analysis_output.median_peakwidth_s = round(peak_properties_df["FWHM"].median(), 2)
    analysis_output.median_rise2080_s = round(
        peak_properties_df["rise2080"].median(), 2
    )
    analysis_output.median_decay8020_s = round(
        peak_properties_df["decay8020"].median(), 2
    )

    # compute median frequency
    (
        analysis_output.median_frequency_mHz,
        analysis_output.active_cells_median_frequency_mHz,
        _,
    ) = find_median_frequency(peak_series, analysis_output.frame_interval)

    return peak_series, peak_properties_df, peak_frame_nonzero, analysis_output


def calculate_periodic_cells(
    peaks: np.ndarray,
    analysis_parameters: AnalysisParameters,
    analysis_output: AnalysisOutput,
) -> Tuple[AnalysisOutput, np.ndarray, np.ndarray]:
    """
    finds the periodic ROIs and compares their number with a random control

    :param peaks: T x n_roi binary peak series, where 1s indicate peaks
    :type peaks: np.ndarray
    :param analysis_parameters: AnalysisParameters object containing the periodicity method and thresholds
    :type analysis_parameters: AnalysisParameters
    :param analysis_output: AnalysisOutput object to be modified with the periodicity results
    :type analysis_output: AnalysisOutput
    :return: returns the modified AnalysisOutput object and the standard deviations and coefficients of variation of the peak intervals of every ROI
    :rtype: Tuple[AnalysisOutput, np.ndarray, np.ndarray]
    """
    results, experimental_stds, experimental_covs = find_oscillating_rois(
        periodicity_method=analysis_parameters.periodicity_method,
        peak_series=peaks,
        std_threshold=analysis_parameters.std_threshold,
        cov_threshold=analysis_parameters.cov_threshold,
        frame_interval=analysis_output.frame_interval,
        n_iter=100,
    )

    analysis_output.std_cutoff = results["std_cutoff"]
    analysis_output.std_quantile = results["std_quantile"]
    analysis_output.cov_cutoff = results["cov_cutoff"]
    analysis_output.cov_quantile = results["cov_quantile"]
    analysis_output.random_below_std = results["random_below_std"]
    analysis_output.experimental_below_std = results["experimental_below_std"]
    analysis_output.random_below_cov = results["random_below_cov"]
    analysis_output.experimental_below_cov = results["experimental_below_cov"]

    return analysis_output, experimental_stds, experimental_covs


def select_intensity_channel(
    image: np.ndarray, channel: int = INTENSITY_CHANNEL
) -> np.ndarray:
    """
    Returns the single channel of a multichannel image that the analysis runs on.

    The projection is computed over every channel, because the segmentation uses all of
    them, but the figures show the channel the intensity traces are extracted from. With
    the default settings that is the non-reference channel, i.e. the calcium indicator
    rather than the stationary label the motion is estimated on.

    :param image: 2D image, or multichannel image of shape (C, Y, X)
    :type image: np.ndarray
    :param channel: index of the channel to select, defaults to the channel the intensities are extracted from
    :type channel: int, optional
    :raises ValueError: if the image is neither 2D nor 3D
    :return: returns the 2D image of the selected channel
    :rtype: np.ndarray
    """
    image = np.asarray(image)

    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(
            f"Expected a 2D image or a multichannel image of shape (C, Y, X), got shape {image.shape}."
        )
    if image.shape[0] == 1:
        return image[0]
    if not 0 <= channel < image.shape[0]:
        warnings.warn(
            f"Channel {channel} does not exist in an image with {image.shape[0]} channels. Using channel 0.",
            UserWarning,
        )
        channel = 0

    return image[channel]


def calculate_synchronicity(
    shortest_path_matrix: np.ndarray,
    peak_series: np.ndarray,
    analysis_parameters: AnalysisParameters,
    analysis_output: AnalysisOutput,
) -> Tuple[AnalysisOutput, np.ndarray, np.ndarray]:
    """
    calculates the synchronicity z-scores for 1st, 2nd and 3rd degree neighbours and updates the analysis output

    :param shortest_path_matrix: ROI x ROI matrix with shortest path lengths between cells n and m
    :type shortest_path_matrix: np.ndarray
    :param peak_series: T x ROI binary matrix with 1s at calcium peak frames
    :type peak_series: np.ndarray
    :param analysis_parameters: AnalysisParameters object containing the time window and other parameters for the analysis
    :type analysis_parameters: AnalysisParameters
    :param analysis_output: AnalysisOutput object to be modified with the synchronicity z-scores
    :type analysis_output: AnalysisOutput
    :return: returns the analysis output, the synchronous peaks matrix of the 1st neighbours in the recording and one example random synchronous peaks matrix
    :rtype: Tuple[AnalysisOutput, np.ndarray, np.ndarray]
    """
    # calculate synchronicity for 1st, 2nd and 3rd degree neighbours
    frame_window = int(analysis_parameters.time_window / analysis_output.frame_interval)
    for neighbour_degree in [3, 2, 1]:
        (
            t_zscore,
            r_zscore,
            true_synchronous_peaks_matrix,
            rand_synchronous_peaks_matrix,
            potential_synchronous_peaks_matrix,
        ) = calculate_synchronicity_zscore(
            shortest_path_matrix,
            peak_series,
            neighbour_degree,
            frame_window * neighbour_degree,
            n_iter=100,
        )
        if neighbour_degree == 1:
            analysis_output.total_coinciding_peaks_1st_neighbour = np.nansum(
                true_synchronous_peaks_matrix
            )
            analysis_output.possible_coinciding_peaks_1st_neighbour = np.nansum(
                potential_synchronous_peaks_matrix
            )
            analysis_output.experiment_1st_neighbour_zscore = round(t_zscore, 2)
            analysis_output.random_1st_neighbour_zscore = round(r_zscore, 2)
        elif neighbour_degree == 2:
            analysis_output.experiment_2nd_neighbour_zscore = round(t_zscore, 2)
            analysis_output.random_2nd_neighbour_zscore = round(r_zscore, 2)
        elif neighbour_degree == 3:
            analysis_output.experiment_3rd_neighbour_zscore = round(t_zscore, 2)
            analysis_output.random_3rd_neighbour_zscore = round(r_zscore, 2)

    return analysis_output, true_synchronous_peaks_matrix, rand_synchronous_peaks_matrix


def create_roi_dataframe(
    peak_series: np.ndarray,
    experimental_stds: np.ndarray,
    experimental_covs: np.ndarray,
    analysis_output: AnalysisOutput,
) -> pd.DataFrame:
    """
    creates a dataframe with ROI properties including peak count, std, cov, and frequency

    :param peak_series: T x n_roi binary matrix with 1s at calcium peak frames
    :type peak_series: np.ndarray
    :param experimental_stds: array with standard deviations of each ROI
    :type experimental_stds: np.ndarray
    :param experimental_covs: array with coefficients of variation of each ROI
    :type experimental_covs: np.ndarray
    :param analysis_output: AnalysisOutput object containing the frame interval
    :type analysis_output: AnalysisOutput
    :return: returns a pandas DataFrame with ROI properties
    :rtype: pd.DataFrame
    """
    roi_df = pd.DataFrame(
        {
            "ROI": pd.Series(dtype="int64"),  # 0-indexed integers
            "number_of_peaks": pd.Series(dtype="int64"),
            "STD_in_seconds": pd.Series(dtype="float64"),
            "CoV": pd.Series(dtype="float64"),
            "STD_is_periodic": pd.Series(dtype="bool"),
            "CoV_is_periodic": pd.Series(dtype="bool"),
            "median_peakinterval_seconds": pd.Series(dtype="float64"),
        }
    )
    _, _, cell_periods = find_median_frequency(
        peak_series, analysis_output.frame_interval
    )

    roi_df["ROI"] = np.arange(peak_series.shape[1])
    roi_df["number_of_peaks"] = np.sum(peak_series, axis=0)
    roi_df["STD_in_seconds"] = experimental_stds
    roi_df["CoV"] = experimental_covs
    roi_df["STD_is_periodic"] = roi_df["STD_in_seconds"] < analysis_output.std_cutoff
    roi_df["CoV_is_periodic"] = roi_df["CoV"] < analysis_output.cov_cutoff
    roi_df["median_peakinterval_seconds"] = cell_periods
    return roi_df


def save_params(
    analysis_output: AnalysisOutput,
    analysis_params: AnalysisParameters,
    overwrite: bool = False,
):
    """
    saves the analysis output and parameters as CSV files to the analysis folder of the video

    :param analysis_output: AnalysisOutput object to be saved
    :type analysis_output: AnalysisOutput
    :param analysis_params: AnalysisParameters object to be saved
    :type analysis_params: AnalysisParameters
    :param overwrite: whether to replace existing files, defaults to False
    :type overwrite: bool, optional
    :raises FileExistsError: if one of the files exists and overwrite is False
    """
    video_path = analysis_output.filepath
    os.makedirs(get_analysis_folder(video_path), exist_ok=True)

    output_file = get_analysis_file(video_path, "analysis_output.csv")
    param_file = get_analysis_file(video_path, "analysis_parameters.csv")

    if not overwrite:
        for f in [output_file, param_file]:
            if os.path.exists(f):
                raise FileExistsError(
                    f"{f} already exists. Enable 'overwrite' to replace."
                )
    analysis_output.to_pandas().to_csv(output_file, index=False)
    analysis_params.to_pandas().to_csv(param_file, index=False)


def save_analysis_results(
    tproj: np.ndarray,
    masks: np.ndarray,
    synchronous_peaks_matrix: np.ndarray,
    analysis_output: AnalysisOutput,
    analysis_params: AnalysisParameters,
    detrended_intensities: np.ndarray,
    peaks: np.ndarray,
    intensities: np.ndarray,
    stds: np.ndarray,
    covs: np.ndarray,
    roi_dataframe: pd.DataFrame = None,
    overwrite: bool = False,
):
    video_path = analysis_output.filepath

    # Save output
    analysis_output.pykrait_version = get_pykrait_version()
    analysis_output.timestamp = np.datetime64("now")
    save_params(analysis_output, analysis_params, overwrite=overwrite)

    # Save arrays
    stds_covs = np.column_stack((stds, covs))
    save_NroisxF(
        stds_covs,
        get_analysis_file(video_path, "stds.csv.zst"),
        header=["STD", "CoV"],
    )
    save_Txnrois(
        peaks,
        analysis_output.frame_interval,
        get_analysis_file(video_path, "peaks.csv.zst"),
    )
    save_Txnrois(
        detrended_intensities,
        analysis_output.frame_interval,
        get_analysis_file(video_path, "detrended_intensities.csv.zst"),
    )
    save_Txnrois(
        intensities,
        analysis_output.frame_interval,
        get_analysis_file(video_path, "raw_intensities.csv.zst"),
    )

    if roi_dataframe is not None:
        roi_dataframe.to_csv(
            get_analysis_file(video_path, "roi_properties.csv"),
            index=False,
        )

    tproj = np.asarray(tproj)
    if tproj.ndim == 3:
        # the projection has the channels of the original or of the motion-corrected video
        tproj = select_intensity_channel(
            tproj,
            channel=intensity_channel_index(
                analysis_params, analysis_output.number_of_channels, tproj.shape[0]
            ),
        )
    create_synchronicity_image(
        tproj=tproj,
        masks=masks,
        synchronous_peaks_matrix=synchronous_peaks_matrix,
        savepath=get_analysis_file(video_path, "synchronicity_on_tproj.svg"),
    )
    create_heatmap_image(
        masks=masks,
        peak_series=peaks,
        savepath=get_analysis_file(video_path, "activity_heatmap.svg"),
    )
