"""Lateral (xy) motion correction for calcium imaging timelapses.

The motion of every frame is estimated on a single reference channel by comparing it
against a template image, and the resulting transform is applied to all channels of
that frame. Three estimation methods are available:

* ``global_shift``: rigid translation from phase cross correlation.
* ``warp_grid``: dense optical flow (DIS), averaged on a coarse mesh grid and
  interpolated back to a smooth dense displacement field.
* ``warp_polynomial_fit``: dense optical flow (DIS), approximated by a 2D polynomial
  surface.

Both warping methods perform a rigid translation first and estimate the residual
non-rigid deformation on top of it.

Adapted from the mesh warp motion correction written by Andres Flores Valle (2025).
"""

import cv2
import dask.array as da
import numpy as np
import os
import threading
import warnings

from concurrent.futures import ThreadPoolExecutor
from typing import Literal, Tuple
from scipy.interpolate import RectBivariateSpline
from skimage.registration import phase_cross_correlation
from tqdm import tqdm

MOTION_CORRECTION_METHODS = ("global_shift", "warp_grid", "warp_polynomial_fit")

# number of frames read from a lazy timelapse in one go; reading frame by frame spends
# far more time on the dask overhead than on the correction itself
FRAME_BLOCK_SIZE = 32


def _to_uint8(
    image: np.ndarray, min_quantile: float = 0.01, max_quantile: float = 0.99
) -> np.ndarray:
    """Converts an image to 8-bit by clipping it between two intensity quantiles.

    The motion estimation backends (DIS optical flow) require 8-bit grayscale input,
    and the quantile clipping makes the estimation robust against intensity outliers.

    :param image: 2D input image of any dtype
    :type image: np.ndarray
    :param min_quantile: lower intensity quantile mapped to 0, defaults to 0.01
    :type min_quantile: float, optional
    :param max_quantile: upper intensity quantile mapped to 255, defaults to 0.99
    :type max_quantile: float, optional
    :return: returns the clipped and rescaled image as 8-bit unsigned integers
    :rtype: np.ndarray
    """
    min_value = np.quantile(image, min_quantile)
    max_value = np.quantile(image, max_quantile)

    if max_value - min_value < 1e-9:  # flat image, nothing to rescale
        return np.zeros(image.shape, dtype=np.uint8)

    scaled = 255 * (image - min_value) / (max_value - min_value)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def _fit_polynomial_surface(
    image: np.ndarray, degree: int, mask: np.ndarray = None
) -> np.ndarray:
    """Fits a 2D polynomial of the given total degree to an image and evaluates it on
    the full pixel grid.

    :param image: 2D image to be approximated, e.g. one component of a flow field
    :type image: np.ndarray
    :param degree: total degree of the polynomial
    :type degree: int
    :param mask: optional 2D mask, only pixels where the mask is non-zero are used for the fit, defaults to None
    :type mask: np.ndarray, optional
    :return: returns the fitted surface with the same shape as the input image
    :rtype: np.ndarray
    """
    n_rows, n_cols = image.shape

    # normalized coordinate grid to keep the design matrix well conditioned
    rows, cols = np.mgrid[0:n_rows, 0:n_cols]
    x = cols.ravel() / max(n_cols - 1, 1)
    y = rows.ravel() / max(n_rows - 1, 1)
    z = image.ravel()

    if mask is not None:
        selected = mask.ravel().astype(bool)
        if not selected.any():
            selected = np.ones_like(z, dtype=bool)
    else:
        selected = np.ones_like(z, dtype=bool)

    n_terms = (degree + 1) * (degree + 2) // 2
    exponents = [
        (i, j) for i in range(degree + 1) for j in range(degree + 1 - i)
    ]  # x**i * y**j with i + j <= degree

    design_matrix = np.empty((x.size, n_terms))
    for index, (i, j) in enumerate(exponents):
        design_matrix[:, index] = (x**i) * (y**j)

    coefficients, _, _, _ = np.linalg.lstsq(
        design_matrix[selected], z[selected], rcond=None
    )

    return (design_matrix @ coefficients).reshape(n_rows, n_cols)


def _smooth_flow_on_grid(flow: np.ndarray, grid_size: int) -> np.ndarray:
    """Averages a dense flow field on a coarse mesh grid and interpolates it back to a
    smooth dense flow field with a bivariate spline.

    The flow is averaged per grid cell and the resulting values are attached to the cell
    centres, which are then used as spline knots.

    :param flow: dense flow field of shape (Y, X, 2), with (x, y) displacements
    :type flow: np.ndarray
    :param grid_size: number of mesh cells along each axis
    :type grid_size: int
    :return: returns the smoothed dense flow field of shape (Y, X, 2)
    :rtype: np.ndarray
    """
    height, width = flow.shape[:2]

    row_edges = np.linspace(0, height, grid_size + 1).round().astype(int)
    col_edges = np.linspace(0, width, grid_size + 1).round().astype(int)

    cell_flow = np.zeros((grid_size, grid_size, 2), dtype=np.float32)
    for j in range(grid_size):
        for i in range(grid_size):
            cell = flow[
                row_edges[j] : row_edges[j + 1], col_edges[i] : col_edges[i + 1]
            ]
            if cell.size > 0:
                cell_flow[j, i] = cell.reshape(-1, 2).mean(axis=0)

    row_centers = (row_edges[:-1] + row_edges[1:]) / 2
    col_centers = (col_edges[:-1] + col_edges[1:]) / 2

    # spline degree is limited by the number of knots along each axis
    spline_degree = min(3, grid_size - 1)

    rows = np.arange(height)
    cols = np.arange(width)
    smoothed_flow = np.empty_like(flow)
    for component in (0, 1):
        spline = RectBivariateSpline(
            row_centers,
            col_centers,
            cell_flow[..., component],
            kx=spline_degree,
            ky=spline_degree,
        )
        smoothed_flow[..., component] = spline(rows, cols)

    return smoothed_flow


def _estimate_dense_flow(
    template_gray: np.ndarray, frame_gray: np.ndarray
) -> np.ndarray:
    """Estimates the dense optical flow between the template and a frame using the
    DIS (Dense Inverse Search) algorithm.

    :param template_gray: 8-bit template image
    :type template_gray: np.ndarray
    :param frame_gray: 8-bit image of the frame to be aligned to the template
    :type frame_gray: np.ndarray
    :return: returns the dense flow field of shape (Y, X, 2) with (x, y) displacements
    :rtype: np.ndarray
    """
    flow_calculator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return flow_calculator.calc(template_gray, frame_gray, None)


def _estimate_global_shift(
    template_gray: np.ndarray, frame_gray: np.ndarray, upsample_factor: int = 1
) -> np.ndarray:
    """Estimates the rigid translation between the template and a frame with phase cross
    correlation.

    :param template_gray: 8-bit template image of shape (Y, X)
    :type template_gray: np.ndarray
    :param frame_gray: 8-bit image of shape (Y, X) used to estimate the translation
    :type frame_gray: np.ndarray
    :param upsample_factor: subpixel precision factor of the cross correlation, defaults to 1
    :type upsample_factor: int, optional
    :return: returns the (row, column) offset that aligns the frame to the template
    :rtype: np.ndarray
    """
    offset, _, _ = phase_cross_correlation(
        template_gray, frame_gray, upsample_factor=upsample_factor
    )
    return offset


def _resample(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Samples an image at the given coordinates, filling everything outside with zeros.

    :param image: 2D image to sample
    :type image: np.ndarray
    :param map_x: column coordinate to sample for every output pixel
    :type map_x: np.ndarray
    :param map_y: row coordinate to sample for every output pixel
    :type map_y: np.ndarray
    :return: returns the resampled image with the same dtype as the input
    :rtype: np.ndarray
    """
    return cv2.remap(
        np.ascontiguousarray(image),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )


def _shift_maps(
    offset: np.ndarray, shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Builds the sampling coordinates of a rigid translation.

    :param offset: (row, column) offset the frame is shifted by
    :type offset: np.ndarray
    :param shape: (height, width) of the frame
    :type shape: Tuple[int, int]
    :return: returns the column and row coordinate maps
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    height, width = shape
    map_x = np.arange(width, dtype=np.float32)[np.newaxis, :] - np.float32(offset[1])
    map_y = np.arange(height, dtype=np.float32)[:, np.newaxis] - np.float32(offset[0])
    return (
        np.broadcast_to(map_x, shape).copy(),
        np.broadcast_to(map_y, shape).copy(),
    )


def _correct_frame(
    template_gray: np.ndarray,
    frame_gray: np.ndarray,
    frame: np.ndarray,
    method: str,
    grid_size: int,
    polynomial_degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Aligns a single multichannel frame to the template with the requested method.

    The rigid translation and the non-rigid flow are composed into a single set of
    sampling coordinates, so every channel is resampled exactly once.

    :param template_gray: 8-bit template image of shape (Y, X)
    :type template_gray: np.ndarray
    :param frame_gray: 8-bit image of shape (Y, X) the motion is estimated on
    :type frame_gray: np.ndarray
    :param frame: multichannel frame of shape (C, Y, X) the correction is applied to
    :type frame: np.ndarray
    :param method: one of "global_shift", "warp_grid" or "warp_polynomial_fit"
    :type method: str
    :param grid_size: number of mesh cells along each axis, only used for "warp_grid"
    :type grid_size: int
    :param polynomial_degree: degree of the polynomial, only used for "warp_polynomial_fit"
    :type polynomial_degree: int
    :return: returns the corrected frame (C, Y, X) and a binary mask (Y, X) that is 0 where the correction moved data outside the field of view
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    shape = template_gray.shape
    ones = np.ones(shape, dtype=np.uint8)

    # every method starts with a rigid translation
    offset = _estimate_global_shift(template_gray, frame_gray)
    map_x, map_y = _shift_maps(offset, shape)
    shift_mask = cv2.remap(
        ones,
        map_x,
        map_y,
        interpolation=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    )

    if method != "global_shift":
        # estimate the residual deformation on the rigidly aligned frame, ignoring the
        # regions that the translation moved out of the field of view
        shifted_gray = _resample(frame_gray, map_x, map_y) * shift_mask
        masked_template_gray = template_gray * shift_mask

        flow = _estimate_dense_flow(masked_template_gray, shifted_gray)
        if method == "warp_grid":
            flow = _smooth_flow_on_grid(flow, grid_size)
        else:
            flow = np.stack(
                [
                    _fit_polynomial_surface(
                        flow[..., component], polynomial_degree, mask=shift_mask
                    )
                    for component in (0, 1)
                ],
                axis=-1,
            ).astype(flow.dtype)

        # compose translation and flow into one set of sampling coordinates
        rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
        flow_map_x = cols + flow[..., 0]
        flow_map_y = rows + flow[..., 1]
        map_x = (flow_map_x - np.float32(offset[1])).astype(np.float32)
        map_y = (flow_map_y - np.float32(offset[0])).astype(np.float32)

        warp_mask = cv2.remap(
            shift_mask,
            flow_map_x.astype(np.float32),
            flow_map_y.astype(np.float32),
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
        )
    else:
        warp_mask = shift_mask

    corrected_frame = np.stack(
        [_resample(frame[channel], map_x, map_y) for channel in range(frame.shape[0])]
    )

    return corrected_frame, warp_mask


def _resolve_reference_channel(reference_channel: int, n_channels: int) -> int:
    """Validates the reference channel against the number of channels of a timelapse.

    Single-channel videos fall back to the only channel they have, so that the default
    reference channel does not have to be changed for them.

    :param reference_channel: index of the channel the motion is estimated on
    :type reference_channel: int
    :param n_channels: number of channels of the timelapse
    :type n_channels: int
    :raises ValueError: if the channel does not exist in a multichannel timelapse
    :return: returns the reference channel that is used
    :rtype: int
    """
    if 0 <= reference_channel < n_channels:
        return reference_channel

    if n_channels == 1 and reference_channel > 0:
        warnings.warn(
            f"reference_channel {reference_channel} does not exist in a single-channel video. "
            "Estimating the motion on channel 0 instead.",
            UserWarning,
        )
        return 0

    raise ValueError(
        f"reference_channel {reference_channel} is out of bounds for number of channels {n_channels}."
    )


def compute_template(
    timelapse, reference_channel: int = 1, template_frames: int = 10
) -> np.ndarray:
    """Computes the alignment template as the average of the first frames of the
    reference channel.

    :param timelapse: 4D timelapse of shape (T, C, Y, X), either a dask or a numpy array
    :type timelapse: da.Array or np.ndarray
    :param reference_channel: index of the channel the motion is estimated on, defaults to 1. Single-channel videos fall back to channel 0.
    :type reference_channel: int, optional
    :param template_frames: number of frames averaged into the template, defaults to 10
    :type template_frames: int, optional
    :raises ValueError: if the timelapse is not 4D or the reference channel does not exist
    :return: returns the 2D template image of shape (Y, X) as float32
    :rtype: np.ndarray
    """
    if timelapse.ndim != 4:
        raise ValueError(
            f"Input timelapse must be a 4D array with shape TCYX and not shape {timelapse.shape}."
        )
    reference_channel = _resolve_reference_channel(
        reference_channel, timelapse.shape[1]
    )
    if template_frames < 1:
        raise ValueError("template_frames must be at least 1.")

    n_frames = min(template_frames, timelapse.shape[0])
    template = np.asarray(timelapse[:n_frames, reference_channel]).mean(axis=0)
    return template.astype(np.float32)


class LateralMotionCorrector:
    """Corrects the lateral (xy) motion of a timelapse, one block of frames at a time.

    The corrected frames are produced by :meth:`iter_frames`, which never holds more than
    one block of frames in memory. This lets timelapses larger than the available memory
    be corrected and written out (or reduced) frame by frame.

    See :func:`correct_lateral_motion` for a description of the parameters. After the
    iteration has finished, ``template`` holds the final template and ``valid_fov`` the
    pixels that stayed inside the field of view in every frame.
    """

    def __init__(
        self,
        timelapse,
        method: Literal[
            "global_shift", "warp_grid", "warp_polynomial_fit"
        ] = "warp_grid",
        reference_channel: int = 1,
        template_frames: int = 10,
        average_frames: int = 3,
        template_filter: float = 0.0,
        grid_size: int = 10,
        polynomial_degree: int = 2,
        min_quantile: float = 0.01,
        max_quantile: float = 0.99,
        template: np.ndarray = None,
        n_workers: int = None,
    ):
        if not isinstance(timelapse, (da.Array, np.ndarray)):
            raise TypeError("Input must be a dask or a numpy array.")
        if timelapse.ndim != 4:
            raise ValueError(
                f"Input timelapse must be a 4D array with shape TCYX and not shape {timelapse.shape}."
            )
        if method not in MOTION_CORRECTION_METHODS:
            raise ValueError(
                f"Unknown motion correction method '{method}'. Available methods are: {list(MOTION_CORRECTION_METHODS)}."
            )

        n_timepoints, n_channels, height, width = timelapse.shape

        reference_channel = _resolve_reference_channel(reference_channel, n_channels)
        if average_frames < 1:
            raise ValueError("average_frames must be at least 1.")
        if not 0 <= template_filter <= 1:
            raise ValueError("template_filter must be between 0 and 1.")
        if method == "warp_grid" and grid_size < 2:
            raise ValueError("grid_size must be at least 2.")
        if method == "warp_polynomial_fit" and polynomial_degree < 1:
            raise ValueError("polynomial_degree must be at least 1.")
        if template_frames < 1:
            raise ValueError("template_frames must be at least 1.")

        self.timelapse = timelapse
        self.method = method
        self.reference_channel = reference_channel
        self.template_frames = template_frames
        self.average_frames = average_frames
        self.template_filter = template_filter
        self.grid_size = grid_size
        self.polynomial_degree = polynomial_degree
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.shape = timelapse.shape
        self.dtype = timelapse.dtype

        # the rolling window spans average_frames frames; an even window reaches one
        # frame further ahead than back
        self._frames_before = (average_frames - 1) // 2
        self._frames_after = average_frames // 2
        self._frame_cache = {}
        self._read_lock = threading.Lock()
        self._pending_blocks = {}
        # a single reader thread, so that reading a block overlaps with correcting the
        # previous one without ever reading the same block twice
        self._reader = ThreadPoolExecutor(max_workers=1)
        self._iterated = False
        self.valid_fov = np.ones((height, width), dtype=np.uint8)

        if template is None:
            n_template_frames = min(template_frames, n_timepoints)
            template = np.mean(
                [
                    self._get_frame(index)[reference_channel]
                    for index in range(n_template_frames)
                ],
                axis=0,
            )
        # copy, the template may be updated while correcting
        self.template = np.array(template, dtype=np.float32)
        if self.template.shape != (height, width):
            raise ValueError(
                f"Template shape {self.template.shape} does not match the frame shape {(height, width)}."
            )

        # with a fixed template the frames are independent and can be corrected in
        # parallel; a filtered template makes every frame depend on the previous one
        self.n_workers = (
            1 if template_filter > 0 else (n_workers or os.cpu_count() or 1)
        )

    def _read_block(self, start: int) -> None:
        """Reads one aligned block of frames and publishes it into the frame cache."""
        end = min(start + FRAME_BLOCK_SIZE, self.shape[0])
        block = np.ascontiguousarray(np.asarray(self.timelapse[start:end]))
        with self._read_lock:
            for offset, frame in enumerate(block):
                self._frame_cache[start + offset] = frame

    def _request_block(self, start: int, wait: bool):
        """Requests an aligned block, either waiting for it or prefetching it in the
        background so that reading overlaps with correcting."""
        if not 0 <= start < self.shape[0]:
            return
        with self._read_lock:
            if start in self._frame_cache:
                return
            future = self._pending_blocks.get(start)
            if future is None:
                future = self._pending_blocks[start] = self._reader.submit(
                    self._read_block, start
                )
        if wait:
            future.result()

    def _request_range(self, start: int, stop: int, wait: bool) -> None:
        """Requests every aligned block covering the frames in [start, stop)."""
        start = max(start, 0)
        stop = min(stop, self.shape[0])
        first = (start // FRAME_BLOCK_SIZE) * FRAME_BLOCK_SIZE
        for block_start in range(first, stop, FRAME_BLOCK_SIZE):
            self._request_block(block_start, wait=wait)

    def _get_frame(self, index: int) -> np.ndarray:
        """Returns a frame, reading its block from the timelapse if it is not cached."""
        if index not in self._frame_cache:
            self._request_block(
                (index // FRAME_BLOCK_SIZE) * FRAME_BLOCK_SIZE, wait=True
            )
        return self._frame_cache[index]

    def _reference_gray(self, timepoint: int) -> np.ndarray:
        """Returns the 8-bit rolling average of the reference channel for a timepoint."""
        window_start = max(timepoint - self._frames_before, 0)
        window_end = min(timepoint + self._frames_after + 1, self.shape[0])
        if window_end - window_start == 1:
            reference_average = self._get_frame(window_start)[self.reference_channel]
        else:
            reference_average = np.mean(
                [
                    self._get_frame(index)[self.reference_channel]
                    for index in range(window_start, window_end)
                ],
                axis=0,
            )
        return _to_uint8(reference_average, self.min_quantile, self.max_quantile)

    def _correct(self, timepoint: int, template_gray: np.ndarray):
        return _correct_frame(
            template_gray=template_gray,
            frame_gray=self._reference_gray(timepoint),
            frame=self._get_frame(timepoint),
            method=self.method,
            grid_size=self.grid_size,
            polynomial_degree=self.polynomial_degree,
        )

    def _drop_stale_frames(self, next_timepoint: int) -> None:
        """Drops the cached frames that no following rolling window needs."""
        oldest_needed = next_timepoint - self._frames_before
        with self._read_lock:
            for cached in [
                index for index in self._frame_cache if index < oldest_needed
            ]:
                del self._frame_cache[cached]
            for pending in [
                start
                for start in self._pending_blocks
                if start + FRAME_BLOCK_SIZE <= oldest_needed
            ]:
                del self._pending_blocks[pending]

    def iter_frames(self, verbose: bool = True):
        """Corrects the timelapse and yields the corrected frames in order.

        A corrector can only be iterated once, as the template and the valid field of view
        are updated while correcting. Create a new corrector to correct the video again.

        :param verbose: if True, shows a progress bar, defaults to True
        :type verbose: bool, optional
        :raises RuntimeError: if the corrector has already been iterated
        :yield: corrected frames of shape (C, Y, X) with the dtype of the input
        :rtype: Iterator[np.ndarray]
        """
        if self._iterated:
            raise RuntimeError(
                "This LateralMotionCorrector has already been iterated. Create a new one to correct the video again."
            )
        self._iterated = True
        n_timepoints = self.shape[0]
        progress = tqdm(
            total=n_timepoints, desc="Correcting lateral motion", disable=not verbose
        )

        def consume(
            timepoint: int, corrected_frame: np.ndarray, valid_mask: np.ndarray
        ):
            if self.template_filter > 0:
                corrected_reference = corrected_frame[self.reference_channel]
                selected = valid_mask > 0
                self.template[selected] = (1 - self.template_filter) * self.template[
                    selected
                ] + self.template_filter * corrected_reference[selected]
            self.valid_fov[valid_mask == 0] = 0
            progress.update(1)

        try:
            if self.n_workers == 1:
                for timepoint in range(n_timepoints):
                    self._request_range(
                        timepoint - self._frames_before,
                        timepoint + self._frames_after + 1,
                        wait=True,
                    )
                    self._request_range(
                        timepoint + self._frames_after + 1,
                        timepoint + self._frames_after + 1 + FRAME_BLOCK_SIZE,
                        wait=False,
                    )
                    template_gray = _to_uint8(
                        self.template, self.min_quantile, self.max_quantile
                    )
                    corrected_frame, valid_mask = self._correct(
                        timepoint, template_gray
                    )
                    consume(timepoint, corrected_frame, valid_mask)
                    self._drop_stale_frames(timepoint + 1)
                    yield corrected_frame
            else:
                # the template is fixed, so a block of frames can be corrected in
                # parallel; the results are consumed in order
                template_gray = _to_uint8(
                    self.template, self.min_quantile, self.max_quantile
                )
                with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    for start in range(0, n_timepoints, FRAME_BLOCK_SIZE):
                        end = min(start + FRAME_BLOCK_SIZE, n_timepoints)
                        # everything this block needs, including the look-ahead of the
                        # rolling window
                        self._request_range(
                            start - self._frames_before,
                            end + self._frames_after,
                            wait=True,
                        )
                        # read the next block while this one is being corrected
                        self._request_range(
                            end + self._frames_after,
                            end + self._frames_after + FRAME_BLOCK_SIZE,
                            wait=False,
                        )
                        results = list(
                            executor.map(
                                lambda timepoint: self._correct(
                                    timepoint, template_gray
                                ),
                                range(start, end),
                            )
                        )
                        for timepoint, (corrected_frame, valid_mask) in zip(
                            range(start, end), results
                        ):
                            consume(timepoint, corrected_frame, valid_mask)
                            yield corrected_frame
                        self._drop_stale_frames(end)
        finally:
            progress.close()
            self._reader.shutdown(wait=False)


def correct_lateral_motion(
    timelapse,
    method: Literal["global_shift", "warp_grid", "warp_polynomial_fit"] = "warp_grid",
    reference_channel: int = 1,
    template_frames: int = 10,
    average_frames: int = 3,
    template_filter: float = 0.0,
    grid_size: int = 10,
    polynomial_degree: int = 2,
    min_quantile: float = 0.01,
    max_quantile: float = 0.99,
    mask_valid_fov: bool = False,
    template: np.ndarray = None,
    verbose: bool = True,
    n_workers: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Corrects the lateral (xy) motion of a timelapse.

    This materializes the whole corrected timelapse in memory. Use
    :class:`LateralMotionCorrector` to stream the corrected frames instead, which is
    required for timelapses that do not fit into memory.

    The motion of every frame is estimated on ``reference_channel`` against a template
    image and the resulting transform is applied to all channels of that frame. To make
    the estimation robust against shot noise, the motion is estimated on a rolling
    average of ``average_frames`` frames of the reference channel, while the correction
    is always applied to the single frame itself.

    :param timelapse: 4D timelapse of shape (T, C, Y, X), either a dask or a numpy array. Dask arrays are read frame by frame, so the input never has to be loaded into memory at once.
    :type timelapse: da.Array or np.ndarray
    :param method: motion estimation method, one of "global_shift" (rigid translation only), "warp_grid" (mesh based optical flow) or "warp_polynomial_fit" (polynomial fit of the optical flow), defaults to "warp_grid"
    :type method: Literal["global_shift", "warp_grid", "warp_polynomial_fit"], optional
    :param reference_channel: index of the channel the motion is estimated on. Works best on a channel with stationary labeling, defaults to 1. Single-channel videos fall back to channel 0.
    :type reference_channel: int, optional
    :param template_frames: number of frames at the beginning of the recording that are averaged into the template, defaults to 10
    :type template_frames: int, optional
    :param average_frames: size of the rolling window of reference frames used for the motion estimation, defaults to 3
    :type average_frames: int, optional
    :param template_filter: factor to update the template with every corrected frame as ``template = (1 - factor) * template + factor * frame``. 0 keeps the template fixed, defaults to 0.0
    :type template_filter: float, optional
    :param grid_size: number of mesh cells along each axis, only used for "warp_grid", defaults to 10
    :type grid_size: int, optional
    :param polynomial_degree: total degree of the fitted polynomial, only used for "warp_polynomial_fit", defaults to 2
    :type polynomial_degree: int, optional
    :param min_quantile: lower intensity quantile used to convert frames to 8-bit for the motion estimation, defaults to 0.01
    :type min_quantile: float, optional
    :param max_quantile: upper intensity quantile used to convert frames to 8-bit for the motion estimation, defaults to 0.99
    :type max_quantile: float, optional
    :param mask_valid_fov: if True, all pixels that left the field of view in any frame are set to 0 in the whole corrected timelapse, defaults to False
    :type mask_valid_fov: bool, optional
    :param template: template image of shape (Y, X) to align to. If None, it is computed from the first ``template_frames`` frames, defaults to None
    :type template: np.ndarray, optional
    :param verbose: if True, shows a progress bar, defaults to True
    :type verbose: bool, optional
    :param n_workers: number of threads used to correct frames in parallel. Only used when the template is fixed (``template_filter`` is 0), defaults to the number of CPUs
    :type n_workers: int, optional
    :raises TypeError: if the input is neither a dask nor a numpy array
    :raises ValueError: if the input is not 4D, if the parameters are out of range or if the template shape does not match the timelapse
    :return: returns the corrected timelapse of shape (T, C, Y, X) with the same dtype as the input, and the final template
    :rtype: Tuple[np.ndarray, np.ndarray]
    """
    corrector = LateralMotionCorrector(
        timelapse,
        method=method,
        reference_channel=reference_channel,
        template_frames=template_frames,
        average_frames=average_frames,
        template_filter=template_filter,
        grid_size=grid_size,
        polynomial_degree=polynomial_degree,
        min_quantile=min_quantile,
        max_quantile=max_quantile,
        template=template,
        n_workers=n_workers,
    )

    corrected = np.zeros(timelapse.shape, dtype=timelapse.dtype)
    for timepoint, corrected_frame in enumerate(corrector.iter_frames(verbose=verbose)):
        corrected[timepoint] = corrected_frame

    if mask_valid_fov:
        corrected *= corrector.valid_fov[None, None, :, :]

    return corrected, corrector.template
