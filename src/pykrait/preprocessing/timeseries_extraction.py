import numpy as np
import pandas as pd
import warnings
import dask.array as da
from tqdm import tqdm

from pykrait.io.frame_cache import iter_frame_blocks
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries
from skimage.morphology import disk
from scipy.ndimage import binary_erosion, distance_transform_edt, label, find_objects
from scipy.sparse.csgraph import shortest_path


def shrink_masks(masks: np.ndarray, shrink_factor: float = 0.7) -> np.ndarray:
    """
    returns a label image with labels shrunk concentrically by a factor

    :param masks: label image of size m x n
    :type label_img: np.ndarray
    :param shrink_factor: factor to shrink label by, defaults to 0.7
    :type shrink_factor: float, optional
    :return: returns the shrunk label image
    :rtype: np.ndarray
    """
    shrunk_img = np.zeros_like(masks)

    # 8-way connectivity in 2D
    kernel = np.ones((3, 3), dtype=bool)

    # every ROI is eroded inside its own bounding box instead of on the full image
    bounding_boxes = find_objects(masks)

    for index, bounding_box in enumerate(bounding_boxes):
        if bounding_box is None:  # label not present in the image
            continue
        label_val = index + 1

        # pad the box so the erosion sees the background around the ROI
        bounding_box = tuple(
            slice(max(axis.start - 1, 0), min(axis.stop + 1, masks.shape[dimension]))
            for dimension, axis in enumerate(bounding_box)
        )
        roi_mask = masks[bounding_box] == label_val
        target = shrunk_img[bounding_box]

        # Compute erosion iterations based on area ratio
        area_original = roi_mask.sum()
        area_target = area_original * shrink_factor

        if area_target < 20:
            warnings.warn(
                f"ROI {label_val} too small to shrink — keeping original.", UserWarning
            )
            target[roi_mask] = label_val
            continue
        radius_original = int(np.sqrt(area_original / np.pi))
        radius_target = int(np.sqrt(area_target / np.pi))
        iterations = max(radius_original - radius_target, 1)

        eroded_mask = binary_erosion(roi_mask, structure=disk(1), iterations=iterations)

        labeled, n = label(eroded_mask, structure=kernel)
        if n == 0:
            # erosion removed the whole ROI, so it is kept unshrunk instead of being lost
            warnings.warn(
                f"ROI {label_val} vanished when shrinking — keeping original.",
                UserWarning,
            )
            target[roi_mask] = label_val
        elif n > 1:
            component_sizes = np.bincount(labeled.ravel())[1:]
            largest_idx = np.argmax(component_sizes) + 1
            target[labeled == largest_idx] = label_val

            warnings.warn(
                f"ROI {label_val} had {n} disconnected components — kept the largest ({component_sizes[largest_idx - 1]} px).",
                UserWarning,
            )
        else:
            target[eroded_mask] = label_val

    return shrunk_img


def extract_mean_intensities(
    dask_timelapse: da.Array,
    masks: np.ndarray,
    verbose: bool = True,
    channel_index: int = 0,
) -> np.ndarray:
    """
    This function computes the mean intensity of each cell across all frames in a timelapse, given a label image that identifies
    the regions.

    Note: If the input timelapse has multiple channels, only the channel at `channel_index` is used for intensity calculations.
    It uses dask to lazily compute the mean intensities for each ROI and timepoint and does not load the entire timelapse into memory.

    :param dask_timelapse: A 4D dask array with order "TCYX", fully loaded into memory.
    :type dask_timelapse: da.Array
    :param masks: A 2D numpy array containing labels for each region as label image.
    :type masks: np.ndarray
    :param verbose: if True, shows the progress bar for the computation.
    :type verbose: bool
    :raises TypeError: If the input is not a dask array.
    :raises ValueError: If the input timelapse is not a 4D array or if the masks do not match the spatial dimensions of the timelapse.

    :return: a T x n_rois numpy array containing the mean intensity of each cell per frame
    :rtype: np.ndarray
    """
    if not isinstance(dask_timelapse, da.Array):
        raise TypeError("Input must be a dask array.")
    if dask_timelapse.ndim != 4:
        raise ValueError(
            "Input timelapse must be a 4D array with shape TCYX and not shape {}.".format(
                dask_timelapse.shape
            )
        )
    if masks.ndim != 2:
        raise ValueError(
            "Masks must be a 2D array representing the label image and not shape {}.".format(
                masks.shape
            )
        )
    if (
        masks.shape[0] != dask_timelapse.shape[2]
        or masks.shape[1] != dask_timelapse.shape[3]
    ):
        raise ValueError(
            "Masks shape (Y,X: {}, {}) must match the spatial dimensions of the timelapse (Y,X: {}, {}).".format(
                masks.shape[0],
                masks.shape[1],
                dask_timelapse.shape[2],
                dask_timelapse.shape[3],
            )
        )
    # check that the timelapse is in TCYX order
    T, C, Y, X = dask_timelapse.shape
    if T < C or T > Y or T > X or C > Y or C > X:
        warnings.warn(
            f"Input array shape of {dask_timelapse.shape} is unusual. Is it in TCYX order?",
            UserWarning,
        )
    if C > 1:
        if channel_index < 0 or channel_index >= C:
            raise ValueError(
                f"channel_index {channel_index} is out of bounds for number of channels {C}."
            )
        warnings.warn(
            f"Input timelapse has multiple channels. Only the channel at index {channel_index} will be used for intensity calculations.",
            UserWarning,
        )

    shrunk_masks = shrink_masks(masks, shrink_factor=0.7)

    # Pixels are grouped by label once, so that the mean of every ROI is a single
    # bincount per frame instead of a full regionprops pass over the label image.
    # Column i always belongs to label i + 1, as the adjacency matrix and the cell
    # properties index ROIs the same way. Labels missing from the masks get NaN columns.
    n_rois = int(masks.max())
    flat_labels = shrunk_masks.ravel()
    pixels_per_label = np.bincount(flat_labels, minlength=n_rois + 1)
    present_labels = np.flatnonzero(pixels_per_label[1:]) + 1  # skip the background
    pixel_counts = pixels_per_label[present_labels]
    if len(present_labels) < n_rois:
        missing = sorted(set(range(1, n_rois + 1)) - set(present_labels.tolist()))
        warnings.warn(
            f"Labels {missing} do not exist in the masks, their intensities are NaN. Relabel the masks to consecutive labels to avoid empty ROIs.",
            UserWarning,
        )

    # the timelapse is streamed in blocks, so it is read once and never held in memory
    frame_bytes = np.dtype(dask_timelapse.dtype).itemsize * Y * X
    block_size = max(1, int(256 * 1024**2 // max(frame_bytes, 1)))

    mean_intensities = np.full((T, n_rois), np.nan, dtype=np.float64)
    with tqdm(
        total=T, desc="Computing mean intensities", disable=not verbose
    ) as progress:
        for start, block in iter_frame_blocks(
            dask_timelapse[:, channel_index], block_size
        ):
            for offset, frame in enumerate(block):
                sums = np.bincount(
                    flat_labels,
                    weights=frame.ravel(),
                    minlength=len(pixels_per_label),
                )[present_labels]
                mean_intensities[start + offset, present_labels - 1] = (
                    sums / pixel_counts
                )
            progress.update(block.shape[0])

    return mean_intensities  # shape: (T, n_rois)


def extract_cell_properties(masks: np.ndarray) -> tuple:
    """
    Returns cell properties (area, axis lengths, perimeter) and positions (centroid coordinates) of each cell in the label image.


    :param masks: input label image
    :type masks: np.ndarray

    :return cell_properties: pandas dataframe of cell properties
    :rtype cell_properties: pd.DataFrame
    :return cell_positions: (nd.array of size len(roi) x 2, np.uint16): pixel positions of every roi
    :rtype cell_positions: np.ndarray
    """

    cell_properties = regionprops_table(
        label_image=masks,
        intensity_image=masks,
        properties=(
            "label",
            "centroid",
            "intensity_max",
            "area",
            "axis_major_length",
            "axis_minor_length",
            "perimeter",
        ),
        separator=",",
    )
    # to get the 0-indexed index array of labels
    cell_indices = cell_properties["intensity_max"].astype(int) - 1

    cell_properties["label"] = cell_indices
    cell_properties = pd.DataFrame(cell_properties)

    ypos = cell_properties["centroid,0"]
    xpos = cell_properties["centroid,1"]

    # maximum pixel value of masks is number of labels/cells
    cell_positions = np.zeros((masks.max(), 2), dtype=np.uint16)
    cell_positions[:, 0] = xpos[cell_indices].astype(np.uint16)
    cell_positions[:, 1] = ypos[cell_indices].astype(np.uint16)

    return cell_properties, cell_positions


def _find_neighbours_of_region(
    masks: np.ndarray, label_val: int, bounding_box: tuple, neighbour_tolerance: int
) -> np.ndarray:
    """Helper function to find the labels of all ROIs within neighbour_tolerance pixels of the outer boundary of a ROI

    The ROI is only processed inside its bounding box, padded by the tolerance and the
    one pixel wide outer boundary, which contains every pixel within reach.

    :param masks: label image containing the ROIs for each cell
    :type masks: np.ndarray
    :param label_val: label of the ROI
    :type label_val: int
    :param bounding_box: bounding box of the ROI as returned by scipy.ndimage.find_objects
    :type bounding_box: tuple
    :param neighbour_tolerance: maximum Euclidean distance in pixels between the outer boundary and a neighbouring ROI
    :type neighbour_tolerance: int
    :return: returns the labels of the neighbouring ROIs, without the background and the ROI itself
    :rtype: np.ndarray
    """
    padding = neighbour_tolerance + 1
    bounding_box = tuple(
        slice(max(axis.start - padding, 0), min(axis.stop + padding, size))
        for axis, size in zip(bounding_box, masks.shape)
    )
    patch = masks[bounding_box]
    boundary = find_boundaries(patch == label_val, mode="outer")
    if not boundary.any():
        return np.array([], dtype=masks.dtype)
    # every pixel within neighbour_tolerance of any boundary pixel, the same pixels as
    # placing a disk of that radius on every boundary pixel, independent of its size
    within_reach = distance_transform_edt(~boundary) <= neighbour_tolerance
    labels = np.unique(patch[within_reach])
    return labels[(labels != 0) & (labels != label_val)]


def get_adjacency_matrix(
    masks: np.ndarray, neighbour_tolerance: int, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the adjacency matrix and shortest path matrix for ROIs within the neighbour_tolerance in the given mask.

    :param masks: 2D label image where the value of every ROI is equal to its index. A pixel value of 0 indicates the image background.
    :type masks: np.ndarray
    :param neighbour_tolerance: maximum distance between two ROIs to still be considered neighbours, in pixels.
    :type neighbour_tolerance: int
    :param verbose: if true, prints a progress bar for adjacency matrix construction, defaults to True
    :type verbose: bool, optional
    :return: returns a ROI x ROI adjacency matrix, where the element at (i, j) is 1 if ROI i and ROI j are neighbours, and 0 otherwise. Also returns a shortest path matrix, where the element at (i, j) is the length of the shortest path between ROI i and ROI j.
    :rtype: tuple[np.ndarray, np.ndarray]

    Two ROIs are neighbours if a pixel of one ROI lies within neighbour_tolerance pixels of the outer boundary of the other.
    The former legacy implementation, which computed the distances between all boundary pixels instead, identified ca. 1.7% more pairs of adjacent ROIs, likely due to edge cases around the kernel radius and the way boundaries are defined here.
    """
    n_labels = masks.max()
    adjacency_matrix = np.zeros((n_labels, n_labels), dtype=np.uint8)
    for index, bounding_box in enumerate(
        tqdm(
            find_objects(masks), desc="Computing adjacency matrix", disable=not verbose
        )
    ):
        if bounding_box is None:  # label not present in the image
            continue
        neighbours = (
            _find_neighbours_of_region(
                masks, index + 1, bounding_box, neighbour_tolerance
            ).astype(np.intp)
            - 1
        )
        adjacency_matrix[index, neighbours] = 1
        adjacency_matrix[neighbours, index] = 1

    shortest_path_matrix = shortest_path(
        adjacency_matrix, method="D", directed=False, unweighted=True
    )
    return adjacency_matrix, np.triu(shortest_path_matrix, k=1)
