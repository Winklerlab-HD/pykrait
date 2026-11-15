"""
Straightforward implementations that the optimized functions of pykrait are tested against.
"""

import warnings

import numpy as np
from skimage.measure import regionprops_table


def _legacy_extract_mean_intensities(
    numpy_timelapse: np.ndarray, masks: np.ndarray
) -> np.ndarray:
    """
    This function computes the mean intensity of each cell across all frames in a timelapse, given a label image that identifies
    the regions. It returns a 2D numpy array where each row corresponds to a cell and each column corresponds to a frame, containing the
    mean intensity of each cell per frame.

    Note: If the input timelapse has multiple channels, only the channel at `channel_index` is used for intensity calculations.

    It expectes the 4D-numpy array with order "TCYX" to be fully loaded into memory. Loading the dataframe into memory and running this function takes more memory and longer than the lazy version.

    :param numpy_timelapse: A 4D-numpy array with order "TCYX", fully loaded into memory.
    :type numpy_timelapse: np.ndarray
    :param masks: A 2D numpy array containing labels for each region as label image.
    :type masks: np.ndarray

    :return: a cell x time numpy ndarray containing the mean intensity of each cell per frame of shape (n_cells, T).
    :rtype: np.ndarray
    """
    if not isinstance(numpy_timelapse, np.ndarray):
        raise TypeError("Input must be a numpy array.")
    if numpy_timelapse.ndim != 4:
        raise ValueError("Input timelapse must be a 4D array with shape TCYX.")
    if masks.ndim != 2:
        raise ValueError("Masks must be a 2D array representing the label image.")
    if (
        masks.shape[0] != numpy_timelapse.shape[2]
        or masks.shape[1] != numpy_timelapse.shape[3]
    ):
        raise ValueError(
            "Masks shape (Y,X: {}, {}) must match the spatial dimensions of the timelapse (Y,X: {}, {}).".format(
                masks.shape[0],
                masks.shape[1],
                numpy_timelapse.shape[2],
                numpy_timelapse.shape[3],
            )
        )

    # check that the timelapse is in TCYX order
    T, C, Y, X = numpy_timelapse.shape
    if T < C or T > Y or T > X or C > Y or C > X:
        warnings.warn(
            f"Input array shape of {numpy_timelapse.shape} is unusual. Is it in TCYX order?",
            UserWarning,
        )
    if C > 1:
        warnings.warn(
            "Input timelapse has multiple channels. Only the first channel (C=0) will be used for intensity calculations.",
            UserWarning,
        )

    video = numpy_timelapse[:, 0, :, :].squeeze()  # Ensure it's 3D (T, Y, X)

    roi_properties = regionprops_table(
        label_image=masks,
        intensity_image=video.transpose(1, 2, 0),
        properties=("label", "intensity_mean"),
        separator=",",
    )
    intensity_array = np.array(list(roi_properties.values())[1:])

    return intensity_array
