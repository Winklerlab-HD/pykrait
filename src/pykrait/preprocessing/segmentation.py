import logging
import dask.array as da
import numpy as np
import warnings
import os
from tqdm import tqdm
from typing import Literal
from skimage import exposure

from pykrait.io.frame_cache import FrameCache, iter_frame_blocks

logger = logging.getLogger(__name__)

# memory budget for one block of frames read from a lazy timelapse. Larger blocks read
# faster, but the whole block is held in memory while it is being accumulated
BLOCK_MEMORY_BUDGET = 256 * 1024**2


def frames_per_block(
    timelapse, memory_budget: int = BLOCK_MEMORY_BUDGET, minimum: int = 1
) -> int:
    """Computes how many frames of a timelapse fit into the given memory budget.

    Used to stream large timelapses that do not fit into memory as a whole.

    :param timelapse: 4D timelapse of shape (T, C, Y, X)
    :type timelapse: da.Array or np.ndarray
    :param memory_budget: memory budget for one block in bytes, defaults to 256 MB
    :type memory_budget: int, optional
    :param minimum: smallest number of frames per block, defaults to 1
    :type minimum: int, optional
    :return: returns the number of frames per block
    :rtype: int
    """
    _, n_channels, height, width = timelapse.shape
    frame_bytes = n_channels * height * width * np.dtype(timelapse.dtype).itemsize
    return max(minimum, int(memory_budget // max(frame_bytes, 1)))


class TemporalProjection:
    """
    Accumulates a projection across the time axis of a timelapse, one block of frames at
    a time, so that the timelapse never has to be held in memory as a whole and is only
    read once.

    The standard deviation is accumulated as the sum and the sum of squares of the frames
    around a fixed offset (the "shifted variance" algorithm). Shifting by a value close to
    the mean keeps the subtraction well conditioned, so the result is accurate to about
    machine precision in float64.

    :param method: projection to compute, either "std" or "sum"
    :type method: Literal["std", "sum"]
    """

    def __init__(self, method: Literal["std", "sum"] = "std"):
        if method not in ("std", "sum"):
            raise ValueError("T-Projection Method must be either 'std' or 'sum'.")
        self.method = method
        self.n_frames = 0
        self._offset = None
        self._sum = None
        self._sum_of_squares = None

    def update(self, block: np.ndarray) -> None:
        """
        Accumulates a block of frames.

        :param block: frames of shape (n, C, Y, X), or a single frame of shape (C, Y, X)
        :type block: np.ndarray
        """
        block = np.asarray(block)
        if block.ndim == 3:
            block = block[np.newaxis]

        if self._sum is None:
            shape = block.shape[1:]
            self._sum = np.zeros(shape, dtype=np.float64)
            if self.method == "std":
                # first frame as offset, it is always close to the mean of the pixel
                self._offset = block[0].astype(np.float64)
                self._sum_of_squares = np.zeros(shape, dtype=np.float64)

        if self.method == "std":
            # accumulated frame by frame, so that a float64 copy of the whole block is
            # never held in memory
            for frame in block:
                centered = np.subtract(frame, self._offset, dtype=np.float64)
                self._sum += centered
                self._sum_of_squares += centered * centered
        else:
            self._sum += block.sum(axis=0, dtype=np.float64)

        self.n_frames += block.shape[0]

    def result(self) -> np.ndarray:
        """
        Returns the raw projection accumulated so far.

        :raises ValueError: if no frame has been accumulated
        :return: returns the projection of shape (C, Y, X)
        :rtype: np.ndarray
        """
        if self._sum is None or self.n_frames == 0:
            raise ValueError("No frames have been accumulated.")

        if self.method == "sum":
            return self._sum

        mean = self._sum / self.n_frames
        variance = self._sum_of_squares / self.n_frames - mean**2
        return np.sqrt(np.maximum(variance, 0.0))  # clip the rounding noise at zero


def finalize_projection(
    raw_projection: np.ndarray,
    normalize: bool = True,
) -> np.ndarray:
    """
    Converts a raw temporal projection into the 16-bit image used for segmentation.

    The projection is accumulated in floating point, so it is rescaled to 16-bit before
    the contrast enhancement, which only accepts integers or floats within [-1, 1]. A sum
    projection in particular runs far outside that range.

    :param raw_projection: raw projection of shape (C, Y, X), as returned by TemporalProjection
    :type raw_projection: np.ndarray
    :param normalize: If True, applies CLAHE normalization, defaults to True
    :type normalize: bool, optional
    :return: 3D "CYX" numpy array in 16-bit
    :rtype: np.ndarray
    """
    raw_projection = np.nan_to_num(raw_projection, nan=0.0)  # Replace NaNs with 0
    # each channel is scaled on its own, so a bright channel does not squash a dim one
    scaled = np.stack([_scale_to_uint16(channel) for channel in raw_projection])

    if not normalize:
        return scaled

    # Enhance contrast per channel (C, Y, X)
    enhanced = np.empty(scaled.shape, dtype=np.uint16)
    for c in range(scaled.shape[0]):
        enhanced[c] = _scale_to_uint16(exposure.equalize_adapthist(scaled[c]))

    return enhanced


def _scale_to_uint16(arr: np.ndarray) -> np.ndarray:
    """Wraps skimage's rescale_intensity to convert an array to 16-bit unsigned integer format.

    :param arr: input array
    :type arr: np.ndarray
    :return: returns the input array scaled to 16-bit unsigned integer format.
    :rtype: np.ndarray
    """
    return exposure.rescale_intensity(arr, out_range="uint16").astype(np.uint16)


def _accumulate_projection(
    timelapse,
    method: Literal["std", "sum"] = "std",
    verbose: bool = False,
    frame_cache: FrameCache = None,
) -> np.ndarray:
    """
    Streams a timelapse block by block and accumulates the temporal projection.

    The timelapse is read exactly once and only one block of frames is held in memory at
    a time, so timelapses larger than the available memory can be projected.

    :param timelapse: 4D timelapse of shape (T, C, Y, X), either a dask or a numpy array
    :type timelapse: da.Array or np.ndarray
    :param method: projection to compute, either "std" or "sum", defaults to "std"
    :type method: Literal["std", "sum"], optional
    :param verbose: whether to show a progress bar, defaults to False
    :type verbose: bool, optional
    :param frame_cache: if given, keeps the frames of one channel while they are read, defaults to None
    :type frame_cache: FrameCache, optional
    :return: returns the raw projection of shape (C, Y, X)
    :rtype: np.ndarray
    """
    n_timepoints = timelapse.shape[0]
    block_size = frames_per_block(timelapse)
    projection = TemporalProjection(method=method)

    description = f"Computing {method.upper()} projection"
    with tqdm(total=n_timepoints, desc=description, disable=not verbose) as progress:
        for start, block in iter_frame_blocks(timelapse, block_size):
            projection.update(block)
            if frame_cache is not None:
                frame_cache.store(start, block)
            progress.update(block.shape[0])

    return projection.result()


def _fast_time_sum(timelapse: da.Array) -> np.ndarray:
    """
    Computes the sum over the time axis (axis=0) of a T,C,Y,X array in a single pass.

    :param timelapse: array with shape (T, C, Y, X)
    :return: array of shape (C, Y, X) with the sum computed over T
    """
    return _accumulate_projection(timelapse, method="sum")


def _fast_time_std(timelapse: da.Array) -> np.ndarray:
    """
    Computes the standard deviation over the time axis (axis=0) of a T,C,Y,X array.

    The timelapse is streamed block by block and reduced in a single pass, so it is read
    only once and never held in memory as a whole.

    :param timelapse: array with shape (T, C, Y, X)
    :return: array of shape (C, Y, X) with std computed over T
    """
    return _accumulate_projection(timelapse, method="std")


def timelapse_projection(
    lazy_timelapse: da.array,
    normalize: bool = True,
    method: Literal["std", "sum"] = "std",
    verbose: bool = True,
    frame_cache: FrameCache = None,
) -> np.ndarray:
    """
    Computes a T-projection of a timelapse to be used for cell segmentation.

    :param lazy_timelapse: 4D dask array representing the timelapse, with shape order "TCYX".
    :type lazy_timelapse: da.array
    :param normalize: If True, applies CLAHE normalization.
    :type normalize: bool, optional
    :param method: method to be used for the projection, either "std" or "sum" for sum, defaults to "std"
    :type method: Literal["std", "sum"], optional, optional
    :param verbose: whether to output progress bar, defaults to True
    :type verbose: bool, optional
    :param frame_cache: if given, keeps the frames of one channel while they are read, so the intensities can be extracted without reading the video again, defaults to None
    :type frame_cache: FrameCache, optional
    :raises TypeError: if input is not a dask array
    :raises ValueError: if input does not have 4 dimensions
    :raises ValueError: if unknown T-proj method
    :return: 3D "CYX" numpy array containing the standard deviation for each pixel across all frames in 16-bit.
    :rtype: np.ndarray
    """

    # Check if the input is a 4D dask array
    if not isinstance(lazy_timelapse, da.Array):
        raise TypeError("Input must be a dask array.")

    if lazy_timelapse.ndim != 4:
        raise ValueError("Input timelapse must be a 4D array with shape TCYX.")

    T, C, Y, X = lazy_timelapse.shape
    if T < C or T > Y or T > X or C > Y or C > X:
        warnings.warn(
            f"Input array shape of {lazy_timelapse.shape} is unusual. Is it in TCYX order?",
            UserWarning,
        )

    if method not in ("std", "sum"):
        raise ValueError("T-Projection Method must be either 'std' or 'sum'.")

    t_proj = _accumulate_projection(
        lazy_timelapse, method=method, verbose=verbose, frame_cache=frame_cache
    )

    return finalize_projection(t_proj, normalize=normalize)


def create_cellpose_segmentation(
    image: np.ndarray, cellpose_model_path: str
) -> np.ndarray:
    """
    Wrapper function to create a label image from an input image using a prespecified cellpose model.

    :param image: image to be processed with cellpose.
    :type image: ndarray
    :param cellpose_model_path: path to the cellpose model file. Can also be cpsam if using cellpose-4 and the standard builtin model.
    :type cellpose_model_path: str

    :return label_img: The generated label image.
    :rtype: np.ndarray

    :raises FileNotFoundError: If the Cellpose model path does not exist.
    :raises ValueError: If the input std image is not a 2D array.
    :raises RuntimeError: If the Cellpose model fails to load with both GPU and CPU backends.
    """

    # check if the model path exists
    if cellpose_model_path != "cpsam" and not os.path.isfile(cellpose_model_path):
        raise FileNotFoundError(
            "Cellpose model not found at {}".format(cellpose_model_path)
        )
    # TODO: check if the model was trained using the same cellpose version

    if image.ndim > 3:
        raise ValueError(
            "Input image must be a 2D (YX) or 3D (CYX) array but got shape {}".format(
                image.shape
            )
        )

    # cellpose is imported here, as importing it takes seconds and most analyses reuse masks
    from cellpose import models

    # setting up the cellpose segmentation, trying GPU first and falling back to CPU if it fails
    try:
        model = models.CellposeModel(gpu=True, pretrained_model=cellpose_model_path)
    except Exception as e:
        logger.warning(f"Error loading Cellpose model wiht GPU backend due to: {e}")
        try:
            model = models.CellposeModel(
                gpu=False, pretrained_model=cellpose_model_path
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load Cellpose model with CPU backend due to: {e}"
            )

    mask, _, _ = model.eval(image)

    if np.max(mask) == 0:
        warnings.warn(
            "The generated mask contains only zeros. No cells were detected.",
            UserWarning,
        )

    return mask
