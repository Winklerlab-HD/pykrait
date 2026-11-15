import logging
import os
import inspect
import threading
import weakref
import aicspylibczi
from dask.base import tokenize as dask_tokenize
import dask.array as da
import numpy as np
import warnings
import re
from bioio import BioImage
import bioio_czi
import bioio_tifffile
import tifffile
import xml.etree.ElementTree as ET
from skimage.measure import find_contours
from skimage.segmentation import relabel_sequential
from scipy.ndimage import find_objects, label

logger = logging.getLogger(__name__)

ALLOWED_IMAGE_EXTENSIONS = [".czi", ".tif", ".tiff"]

# page stores of the lazily read TIFFs by file, see release_tiff_handles
_RELEASE_TIFF_HANDLES = os.name == "nt"
_tiff_stores = {}
_tiff_stores_lock = threading.Lock()


def _store_key(file_path: str) -> str:
    return os.path.normcase(os.path.abspath(file_path))


def _register_tiff_store(file_path: str, store) -> None:
    """Remembers the page store of a lazily read TIFF, without keeping it alive."""
    key = _store_key(file_path)
    store_id = id(store)

    def forget(_):
        with _tiff_stores_lock:
            stores = _tiff_stores.get(key, {})
            stores.pop(store_id, None)
            if not stores:
                _tiff_stores.pop(key, None)

    with _tiff_stores_lock:
        _tiff_stores.setdefault(key, {})[store_id] = weakref.ref(store, forget)


def release_tiff_handles(file_path: str) -> None:
    """Closes the file handles the lazily read arrays of a TIFF keep open.

    Windows cannot replace or delete a file that is open, so this is called before a
    TIFF that was read with `load_timelapse_lazy` is overwritten. The arrays stay usable,
    they open the file again when they are read. Elsewhere it does nothing: an open file
    can be replaced there, and the arrays keep reading the file they were created from
    instead of a replacement with another layout.

    :param file_path: path of the TIFF
    :type file_path: str
    """
    if not _RELEASE_TIFF_HANDLES:
        return
    with _tiff_stores_lock:
        refs = list(_tiff_stores.get(_store_key(file_path), {}).values())
    for ref in refs:
        store = ref()
        if store is not None:
            store.close()


def load_timelapse_lazy(
    file_path: str,
) -> list[da.Array, float, float, float]:
    """Load a timelapse image file lazily using AICSImageIO, returning a Dask array of the image data along with the frame interval and pixel sizes.

    Currently supports CZI and TIF/TIFF files. The image data is returned as a Dask array with shape (T, C, Y, X) where T is time, C is channels, Y is height, and X is width. The Z dimension is dropped if it has only one slice.

    :param file_path: input file path to the timelapse image file
    :type file_path: str
    :raises TypeError: if filepath is not a string
    :raises FileNotFoundError: if the file does not exist
    :raises ValueError: if the extension is not supported
    :raises ValueError: if the image does not have the expected 5D shape (TCZYX)
    :raises ValueError: if the image has more than one Z slice and Z cannot be automatically dropped
    :return: returns a list of a Dask array containing the image data, the frame interval in seconds, and pixel sizes in micrometers (Y, X)
    :rtype: list[da.Array, float, float, float]
    """
    if not isinstance(file_path, str):
        raise TypeError("file_path must be a string")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Image to load has the unsupported file format '{ext}'. Supported file formats are: {ALLOWED_IMAGE_EXTENSIONS}"
        )

    # Load with BioImage using Dask
    if ext == ".czi":
        logger.info("Reading the metadata with bioio_czi (aicspylibczi)")
        img = BioImage(
            file_path,
            reconstruct_mosaic=False,
            reader=bioio_czi.Reader,
            **_czi_reader_kwargs(),
        )
    elif ext in [".tif", ".tiff"]:
        logger.info("Reading the metadata with bioio_tifffile")
        img = BioImage(
            file_path, reconstruct_mosaic=False, reader=bioio_tifffile.Reader
        )
    # extract the relevant metadata
    pixel_sizes = img.physical_pixel_sizes  # Units are in micrometers (µm)
    # files without calibration report None for the individual axes
    # kept unrounded, e.g. 0.1625 µm would otherwise be off by 1.5%
    y_um = float(pixel_sizes.Y) if pixel_sizes and pixel_sizes.Y else None
    x_um = float(pixel_sizes.X) if pixel_sizes and pixel_sizes.X else None
    if ext in (".tif", ".tiff") and not _tiff_has_physical_unit(file_path):
        # an uncalibrated TIFF stores a resolution of 1 without a unit, which bioio
        # reports as a pixel size of 1 µm
        y_um = x_um = None

    frame_interval = None

    # accessing frame interval from tif/tiff meadata
    if ext == ".tiff" or ext == ".tif":
        try:
            match = re.search(r"finterval=([0-9.]+(?:[eE][-+]?[0-9]+)?)", img.metadata)
            if match is not None:
                frame_interval = float(match.group(1))
            elif img.metadata.lstrip().startswith("<"):
                # OME-XML; an ImageJ description without a frame interval has none
                root = ET.fromstring(img.metadata)
                deltas = []
                for plane in root.findall(".//{*}Plane"):
                    dt = plane.attrib.get("DeltaT")
                    if dt is not None:
                        deltas.append(float(dt))

                if len(deltas) >= 2:
                    # Compute median interval between frames
                    delta_ts_unique = sorted(set(deltas))
                    frame_interval = float(np.median(np.diff(delta_ts_unique)))
                else:
                    # Some files store TimeIncrement directly
                    increment = root.find(".//{*}Pixels")
                    if increment is not None and "TimeIncrement" in increment.attrib:
                        frame_interval = float(increment.attrib["TimeIncrement"])
        except Exception as e:
            warnings.warn(
                f"Failed to extract frame interval from metadata: {type(e).__name__}: {e}",
                UserWarning,
            )
    elif ext == ".czi":
        try:
            root = img.metadata
            increment_elem = root.find(".//T/Positions/Interval/Increment")

            if increment_elem is not None and increment_elem.text:
                frame_interval = float(increment_elem.text)
            else:
                frame_interval = img.time_interval.total_seconds()
        except Exception as e:
            warnings.warn(
                f"Failed to extract frame interval from metadata: {type(e).__name__}: {e}",
                UserWarning,
            )
    dask_img = img.dask_data  # TCZYX
    frame_reader = "bioio_czi (aicspylibczi)" if ext == ".czi" else "bioio_tifffile"

    if ext in [".tif", ".tiff"]:
        # reading TIFF pages through bioio costs far more per frame than reading them
        # directly, so the data is swapped for a page-backed array where possible
        faster_img = _tiff_dask_array(file_path, expected_shape=dask_img.shape)
        if faster_img is not None:
            dask_img = faster_img
            frame_reader = "tifffile, page by page"
    elif ext == ".czi":
        # the same for CZI files, which bioio reads one plane per dask task
        faster_img = _czi_dask_array(file_path, reference=dask_img)
        if faster_img is not None:
            dask_img = faster_img
            frame_reader = "pylibCZIrw"
    logger.info(f"Reading the frames with {frame_reader}")

    if dask_img.ndim != 5:
        raise ValueError(f"Expected 5D image (TCZYX), but got shape {dask_img.shape}")

    T, C, Z, Y, X = dask_img.shape

    if T > 1 and Z == 1:
        dask_img = dask_img[:, :, 0, :, :]  # shape: (T, C, Z, Y, X)
    elif len([dim for dim in dask_img.shape if dim != 1]) == 3:  #
        logger.info(
            "Image has three non-singleton dimensions, presuming order is (T, Y, X)."
        )
        dask_img = _reorder_dask_array(dask_img)  # reorder to (T, C, Y, X)
        dask_img = dask_img[:, :, 0, :, :]  # shape: (T, C, Z, Y, X)
    else:
        raise ValueError(
            f"Cannot drop Z dimension automatically: Z={Z}. Consider handling it explicitly. Image shape: {dask_img.shape}"
        )

    logger.info(f"Returning lazy array of shape {dask_img.shape} (T, C, Y, X)")
    return dask_img, frame_interval, y_um, x_um


def _tiff_has_physical_unit(file_path: str) -> bool:
    """
    Checks whether a TIFF declares a physical unit for its pixel size: an ImageJ unit
    other than pixels, a physical size in its OME-XML, or an inch or centimetre
    resolution unit. TIFFs without calibration store a resolution of 1 without unit.

    :param file_path: path of the TIFF file
    :type file_path: str
    :return: returns True if the pixel size of the TIFF is calibrated
    :rtype: bool
    """
    try:
        with tifffile.TiffFile(file_path) as tif:
            if tif.is_ome and "PhysicalSizeX" in (tif.ome_metadata or ""):
                return True
            imagej_metadata = tif.imagej_metadata
            if imagej_metadata is not None:
                unit = str(imagej_metadata.get("unit", "")).strip().lower()
                return unit not in ("", "pixel", "pixels")
            page = tif.pages.first
            unit_tag = page.tags.get("ResolutionUnit")
            return unit_tag is not None and unit_tag.value in (
                tifffile.RESUNIT.INCH,
                tifffile.RESUNIT.CENTIMETER,
            )
    except Exception as e:
        logger.warning(
            f"Could not check the pixel size unit of {file_path}: {type(e).__name__}: {e}"
        )
        return True  # keep the pixel size bioio reports


def _czi_reader_kwargs() -> dict:
    """Returns the keyword arguments that select the aicspylibczi backend of bioio_czi.

    bioio-czi < 4 defaults to pylibczirw and needs ``use_aicspylibczi=True`` to read with
    aicspylibczi, whose metadata the frame interval lookup relies on. bioio-czi >= 4 only
    has the aicspylibczi backend and rejects the argument.

    :return: keyword arguments to pass to bioio_czi.Reader
    :rtype: dict
    """
    if "use_aicspylibczi" in inspect.signature(bioio_czi.Reader.__init__).parameters:
        return {"use_aicspylibczi": True}
    return {}


def _tiff_dask_array(file_path: str, expected_shape: tuple) -> da.Array:
    """Builds a Dask array that reads a TIFF page by page.

    Reading a TIFF through bioio goes through a large per-chunk overhead, which dominates
    the runtime of every stage that streams a timelapse. Backing the array by the pages of
    the file directly is orders of magnitude faster while returning the same data.

    :param file_path: path to the TIFF file
    :type file_path: str
    :param expected_shape: (T, C, Z, Y, X) shape the array must have to be used
    :type expected_shape: tuple
    :return: returns the page-backed Dask array, or None if the file does not map cleanly onto the expected shape
    :rtype: da.Array
    """
    try:
        with tifffile.TiffFile(file_path) as tiff:
            axes = tiff.series[0].axes

        if any(axis not in "TCZYX" for axis in axes) or len(set(axes)) != len(axes):
            return None  # unusual layout, leave it to bioio

        store = tifffile.imread(file_path, aszarr=True)
        _register_tiff_store(file_path, store)
        dask_img = da.from_zarr(store)

        # add the axes the file does not have and sort them into (T, C, Z, Y, X)
        present = list(axes)
        for axis in "TCZYX":
            if axis not in present:
                dask_img = dask_img[np.newaxis]
                present.insert(0, axis)
        dask_img = dask_img.transpose([present.index(axis) for axis in "TCZYX"])

        if dask_img.shape != tuple(expected_shape):
            return None  # bioio reads this file differently, do not second-guess it

        return dask_img
    except Exception as e:
        warnings.warn(
            f"Could not read {file_path} page by page, falling back to the default reader: {type(e).__name__}: {e}",
            UserWarning,
        )
        return None


# pixel types of CZI files the direct reader supports, with their numpy dtype
_CZI_PIXEL_TYPES = {"Gray8": np.uint8, "Gray16": np.uint16, "Gray32Float": np.float32}
# frames per dask chunk of the direct CZI reader. One frame per chunk, so that showing
# a single frame in the GUI does not read and decode its neighbours as well
_CZI_FRAMES_PER_CHUNK = 1


class _CziPlanes:
    """Array-like (T, C, Z, Y, X) view of a CZI file that reads its planes with pylibCZIrw.

    One reader is kept open and shared by all dask tasks, guarded by a lock, so the file
    is read sequentially, which suits slow drives.
    """

    def __init__(self, file_path: str, shape: tuple, dtype, starts: dict):
        from pylibCZIrw import czi as pyczi

        self._reader = pyczi.CziReader(file_path)
        self._lock = threading.Lock()
        self._starts = starts  # first index of T, C and Z in the file
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self.ndim = 5

    def __getitem__(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        key = key + (slice(None),) * (self.ndim - len(key))
        # planes to read, the Y and X selection is applied to the read planes
        indices = [
            np.arange(size)[axis_key] for axis_key, size in zip(key[:3], self.shape[:3])
        ]
        squeeze = tuple(0 if np.ndim(index) == 0 else slice(None) for index in indices)
        indices = [np.atleast_1d(index) for index in indices]
        out = np.empty(
            tuple(len(index) for index in indices) + self.shape[3:], self.dtype
        )
        with self._lock:
            for i, t in enumerate(indices[0]):
                for j, c in enumerate(indices[1]):
                    for k, z in enumerate(indices[2]):
                        plane = {
                            "T": self._starts["T"] + int(t),
                            "C": self._starts["C"] + int(c),
                            "Z": self._starts["Z"] + int(z),
                        }
                        out[i, j, k] = self._reader.read(plane=plane)[..., 0]
        return out[squeeze + key[3:]]

    def close(self):
        self._reader.close()


def _czi_dask_array(file_path: str, reference: da.Array) -> da.Array:
    """Builds a Dask array that reads a CZI file with pylibCZIrw.

    bioio reads every plane of a CZI in its own dask task, whose overhead makes reading
    several times slower than the file can be decoded. Reading the planes directly
    returns the same data. Files the direct reader might read differently (several scenes,
    mosaics, colour images) and files whose first and last frame do not match bioio's
    are left to bioio.

    :param file_path: path to the CZI file
    :type file_path: str
    :param reference: (T, C, Z, Y, X) array returned by bioio for the same file
    :type reference: da.Array
    :return: returns the directly read Dask array, or None if the file is left to bioio
    :rtype: da.Array
    """
    try:
        from pylibCZIrw import czi as pyczi
    except ImportError:
        return None  # e.g. Intel Macs, which have no pylibCZIrw wheel

    try:
        with open(file_path, "rb") as czi_file:
            if aicspylibczi.CziFile(czi_file).is_mosaic():
                return None

        with pyczi.open_czi(file_path) as reader:
            pixel_types = set(reader.pixel_types.values())
            box = reader.total_bounding_box
            rectangle = reader.total_bounding_rectangle
            scenes = reader.scenes_bounding_rectangle
        if len(pixel_types) != 1 or next(iter(pixel_types)) not in _CZI_PIXEL_TYPES:
            return None
        dtype = _CZI_PIXEL_TYPES[next(iter(pixel_types))]
        if len(scenes) > 1 or any(scene != rectangle for scene in scenes.values()):
            return None

        sizes = {
            axis: box.get(axis, (0, 1))[1] - box.get(axis, (0, 1))[0] for axis in "TCZ"
        }
        shape = (sizes["T"], sizes["C"], sizes["Z"], rectangle.h, rectangle.w)
        if shape != tuple(reference.shape) or np.dtype(dtype) != reference.dtype:
            return None

        planes = _CziPlanes(
            file_path,
            shape,
            dtype,
            starts={axis: box.get(axis, (0, 1))[0] for axis in "TCZ"},
        )
        dask_img = da.from_array(
            planes,
            chunks=(_CZI_FRAMES_PER_CHUNK, 1) + shape[2:],
            name=f"czi-planes-{dask_tokenize(file_path, os.path.getmtime(file_path))}",
            asarray=True,
            fancy=False,
            meta=np.empty((0,) * 5, dtype=dtype),
        )

        # the first and last frame must be the ones bioio reads
        for t in (0, shape[0] - 1):
            if not np.array_equal(np.asarray(dask_img[t]), np.asarray(reference[t])):
                planes.close()
                return None
        return dask_img
    except Exception as e:
        warnings.warn(
            f"Could not read {file_path} directly, falling back to the default reader: {type(e).__name__}: {e}",
            UserWarning,
        )
        return None


def _reorder_dask_array(
    dask_img: da.Array,
) -> da.Array:
    """Reorders a Dask array to the specified order.

    :param dask_array: input Dask array
    :type dask_array: da.Array
    :return: reordered Dask array
    :rtype: da.Array
    """
    shape = dask_img.shape
    non_1_axes = [i for i, dim in enumerate(shape) if dim != 1]

    if len(non_1_axes) < 3:
        raise ValueError(f"Expected at least 3 non-1 dimensions in {shape}")

    # Identify T, Y, X from non-1 dimensions
    non1_indices = [i for i, d in enumerate(dask_img.shape) if d != 1]

    T_axis = non1_indices[0]  # First non-singleton = T
    Y_axis = non1_indices[-2]  # Second last non-singleton = Y
    X_axis = non1_indices[-1]  # Last non-singleton = X

    # Fill in dummy axes for C and Z
    all_axes = list(range(5))
    used_axes = {T_axis, Y_axis, X_axis}
    unused_axes = [a for a in all_axes if a not in used_axes]

    # Fix order to (T, C, Z, Y, X)
    axis_map = {
        "T": T_axis,
        "C": unused_axes[0],
        "Z": unused_axes[1],
        "Y": Y_axis,
        "X": X_axis,
    }

    perm = [axis_map[k] for k in ["T", "C", "Z", "Y", "X"]]

    return dask_img.transpose(*perm)


def read_image(
    file_path: str,
) -> np.ndarray:
    """Reads a image from a file and returns it as a Dask array.

    :param file_path: path to the image file
    :type file_path: str
    :return: Dask array of the image
    :rtype: da.Array
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        img = BioImage(file_path, reconstruct_mosaic=True).dask_data.compute()
        return img[
            0, 0, 0, :, :
        ]  # Assuming the label image is single-channel, return as (Y, X)
    except Exception as e:
        raise ValueError(
            f"Failed to read label image from {file_path}: {type(e).__name__}: {e}"
        )
        return None


def read_label_image(
    file_path: str,
) -> np.ndarray:
    """Reads a label image from a file and returns it as a Dask array.

    :param file_path: path to the label image file
    :type file_path: str
    :return: Dask array of the label image
    :rtype: da.Array
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        img = BioImage(file_path, reconstruct_mosaic=True).dask_data.compute()
        return clean_label_image(img[0, 0, 0, :, :])
    except Exception as e:
        raise ValueError(
            f"Failed to read label image from {file_path}: {type(e).__name__}: {e}"
        )


def read_projection(
    file_path: str,
) -> np.ndarray:
    """Reads a temporal projection saved by pykrait, keeping all of its channels.

    :param file_path: path to the projection image
    :type file_path: str
    :return: projection of shape (C, Y, X)
    :rtype: np.ndarray
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return np.asarray(
        BioImage(file_path, reconstruct_mosaic=True).dask_data[0, :, 0, :, :]
    )


def clean_label_image(
    masks: np.ndarray,
) -> np.ndarray:
    """Keeps only the largest connected component of every ROI and relabels the ROIs to 1..N.

    The analysis indexes ROI n by label n + 1, so gaps in the labels, e.g. from deleted
    ROIs, would shift every ROI after the gap.

    :param masks: label image where 0 is background
    :type masks: np.ndarray
    :return: label image with one connected component per ROI and consecutive labels
    :rtype: np.ndarray
    """
    clean_masks = np.zeros_like(masks)
    roi_labels = np.unique(masks)
    roi_labels = roi_labels[roi_labels != 0]

    # 8-way connectivity in 2D
    kernel = np.ones((3, 3), dtype=bool)

    # every ROI is processed inside its bounding box instead of on the full image; the
    # components are numbered in the same order there, so the result is the same
    bounding_boxes = _bounding_boxes(masks)

    for label_val in roi_labels:
        bounding_box = bounding_boxes[label_val]
        roi_mask = masks[bounding_box] == label_val
        labeled, n = label(roi_mask, structure=kernel)

        # if mask is not connected
        if n > 1:
            component_sizes = np.bincount(labeled.ravel())[1:]
            largest_idx = np.argmax(component_sizes) + 1
            clean_masks[bounding_box][labeled == largest_idx] = label_val

            warnings.warn(
                f"ROI {label_val} had {n} disconnected components — kept the largest ({component_sizes[largest_idx - 1]} px).",
                UserWarning,
            )
        else:
            clean_masks[bounding_box][roi_mask] = label_val

    if len(roi_labels) and roi_labels[-1] != len(roi_labels):
        warnings.warn(
            f"Label image has {len(roi_labels)} ROIs but labels up to {roi_labels[-1]}. Relabeling them to 1..{len(roi_labels)}.",
            UserWarning,
        )
        clean_masks = relabel_sequential(clean_masks)[0]
    return clean_masks


def _bounding_boxes(label_img: np.ndarray, padding: int = 0) -> dict:
    """
    Returns the bounding box of every label of a label image, padded by `padding` pixels
    within the image.

    :param label_img: label image with non-negative integer labels
    :type label_img: np.ndarray
    :param padding: number of pixels to add on every side, defaults to 0
    :type padding: int, optional
    :return: returns a dictionary from label to its bounding box as a tuple of slices
    :rtype: dict
    """
    boxes = {}
    for index, box in enumerate(find_objects(np.asarray(label_img, dtype=np.int64))):
        if box is not None:
            boxes[index + 1] = tuple(
                slice(max(axis.start - padding, 0), min(axis.stop + padding, size))
                for axis, size in zip(box, label_img.shape)
            )
    return boxes


def get_pixel_contour_from_label_img(
    label_img: np.ndarray, orig_shape: tuple = (1, 1), target_shape: tuple = (1, 1)
) -> list:
    """
    Generates pixel contours from a labeled image, scaled to a target shape.

    :param label_img: a label image where each pixel is labeled with an integer representing the ROI it belongs to
    :type label_img: np.ndarray
    :param orig_shape: original shape of the image (height, width)
    :type orig_shape: tuple
    :param target_shape: target shape to which the contours should be scaled (height, width)
    :type target_shape: tuple
    :return: returns a list of polygons representing the scaled contours of each ROI
    :rtype: list
    """
    roi_polygons = []
    orig_height, orig_width = orig_shape
    target_height, target_width = target_shape
    scale_y = target_height / orig_height
    scale_x = target_width / orig_width
    bounding_boxes = _bounding_boxes(label_img, padding=1)
    for index in range(1, label_img.max() + 1):  # skip background (0)
        bounding_box = bounding_boxes.get(index)
        if bounding_box is None:
            continue  # label not present, find_contours would not find a contour
        # traced inside the bounding box, padded so the contour closes as on the full
        # image, and moved back to image coordinates
        contours = find_contours(label_img[bounding_box] == index, 0.5)
        if contours:
            contour = contours[0]
            y, x = contour.T
            y = y + bounding_box[0].start
            x = x + bounding_box[1].start
            y, x = y * scale_y, x * scale_x  # rescales coordinates to target shape
            polygon = list(zip(x, y))
            roi_polygons.append(polygon)
    return roi_polygons
