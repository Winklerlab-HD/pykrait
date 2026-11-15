import logging
import os
import fnmatch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from importlib.metadata import version

logger = logging.getLogger(__name__)


ANALYSIS_FOLDER_PREFIX = "Analysis_"


def get_analysis_stem(video_path: str) -> str:
    """Returns the name a video's analysis files are derived from.

    The extension is dropped, including the ``.ome`` of an ``.ome.tif`` file, so that an
    intermediate result such as ``video_motion_corrected.ome.tif`` does not produce files
    named ``video_motion_corrected.ome_...``.

    :param video_path: path to the video file
    :type video_path: str
    :return: returns the file name without its extensions
    :rtype: str
    """
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if stem.lower().endswith(".ome"):
        stem = stem[: -len(".ome")]
    return stem


def get_analysis_folder(video_path: str) -> str:
    """Returns the folder the analysis results of a video belong in.

    Results are normally written to an ``Analysis_<video_name>`` folder next to the video.
    A video that already lies inside an analysis folder keeps using that folder, so that
    opening an intermediate result (a motion-corrected video, a projection, ...) does not
    nest another analysis folder inside the existing one.

    :param video_path: path to the video file
    :type video_path: str
    :return: returns the path of the analysis folder
    :rtype: str
    """
    parent_dir = os.path.dirname(os.path.abspath(video_path))

    if os.path.basename(parent_dir).startswith(ANALYSIS_FOLDER_PREFIX):
        return parent_dir

    return os.path.join(
        parent_dir, f"{ANALYSIS_FOLDER_PREFIX}{get_analysis_stem(video_path)}"
    )


def get_analysis_file(video_path: str, suffix: str, analysis_folder: str = None) -> str:
    """Returns the path of one of a video's analysis files, e.g. ``<stem>_cp_masks.ome.tif``.

    :param video_path: path to the video file
    :type video_path: str
    :param suffix: name of the file after the video's stem, e.g. "cp_masks.ome.tif"
    :type suffix: str
    :param analysis_folder: folder of the analysis files, defaults to the video's analysis folder
    :type analysis_folder: str, optional
    :return: returns the path of the analysis file
    :rtype: str
    """
    if analysis_folder is None:
        analysis_folder = get_analysis_folder(video_path)
    return os.path.join(analysis_folder, f"{get_analysis_stem(video_path)}_{suffix}")


def _matches_pattern(filename: str, pattern: str) -> bool:
    """Checks a filename against a filter pattern.

    Patterns containing a shell-style wildcard ("*", "?" or "[]") have to match the
    whole filename, all other patterns are matched as a substring. The matching is
    always case-sensitive, independent of the file system.

    :param filename: name of the file, without its directory
    :type filename: str
    :param pattern: filter pattern, a substring or a shell-style wildcard pattern
    :type pattern: str
    :return: returns True if the filename matches the pattern
    :rtype: bool
    """
    if any(wildcard in pattern for wildcard in "*?["):
        return fnmatch.fnmatchcase(filename, pattern)
    return pattern in filename


def get_files_from_folder(
    folder: str,
    extension: str = ".czi",
    filter_string: str = None,
    exclude_string: str = None,
    recursive: bool = False,
) -> list[str]:
    """returns all files from a folder with specified extensions.

    Analysis folders created by pyKrait (`Analysis_<video_name>`) below `folder` are never
    searched, so that a recursive run does not pick up its own results.

    :param folder: path to the folder
    :type folder: str
    :param extensions: str of file extension (default is ".czi")
    :type extensions: str
    :param filter_string: if given, only files whose name matches this pattern are returned. Shell-style wildcards are supported ("*_timeseries.czi"), any other pattern is matched as a substring ("timeseries"), defaults to None
    :type filter_string: str, optional
    :param exclude_string: if given, files whose name matches this pattern are skipped. Uses the same matching as `filter_string`, defaults to None
    :type exclude_string: str, optional
    :param recursive: whether to also search all subfolders, defaults to False
    :type recursive: bool, optional
    :return: sorted list of file paths with the specified extensions
    :rtype: list[str]
    """
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Provided path is not a directory: {folder}")

    root = Path(folder)
    entries = root.rglob("*") if recursive else root.glob("*")

    files = []
    for entry in entries:
        if not entry.is_file() or entry.name.startswith("."):
            continue
        if entry.suffix.lower() != extension.lower():
            continue
        # skip results of previous analyses; only the folders below the searched folder
        # count, so an experiment stored in e.g. "Analysis_2026/" is still searched
        if any(
            part.startswith("Analysis_")
            for part in entry.relative_to(root).parent.parts
        ):
            continue
        if filter_string is not None and not _matches_pattern(
            entry.name, filter_string
        ):
            continue
        if exclude_string is not None and _matches_pattern(entry.name, exclude_string):
            continue
        files.append(str(entry))

    return sorted(files)


def get_pykrait_version() -> str:
    """returns the current version of pykrait, or None if it cannot be determined (e.g. the package is not installed)."""
    try:
        return version("pykrait")
    except Exception as e:
        logger.warning(f"Could not retrieve pykrait version: {e}")
        return None


def _make_parent_dir(filepath: str) -> None:
    """creates the parent directory of a file if it does not exist yet. Bare filenames (in the current directory) are left alone.

    :param filepath: path to the file
    :type filepath: str
    """
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def save_Txnrois(array: np.ndarray, frame_interval: float, filepath: str) -> None:
    """
    Saves a T x n_roi array to a CSV file with a header to denote the ROIs and a time index

    :param array: T x n_roi array to save
    :type array: np.ndarray
    :param filename: name of the output file
    :type filename: str
    :param frame_interval: frame interval in seconds
    :type frame_interval: float
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")

    if array.ndim != 2:
        raise ValueError("Input array must be 2D (T x n_roi)")

    n_frames, n_rois = array.shape
    # rounding removes floating point noise, 6 decimals keep sub-millisecond frame intervals distinct
    timepoints = np.round((np.arange(n_frames) * frame_interval), 6)
    columns = [f"ROI_{i}" for i in range(n_rois)]

    df = pd.DataFrame(array, columns=columns, index=timepoints)
    df.index.name = "Time (s)"
    _make_parent_dir(filepath)
    df.to_csv(filepath, index=True, compression="zstd")


def read_Txnrois(filepath: str, n_frames: int = None, n_rois: int = None) -> np.ndarray:
    """
    loads a T x n_roi array to a CSV file with a header to denote the ROIs and a time index

    :param filepath: path to the CSV file
    :type filepath: str
    :param n_frames: number of frames in the timelapse
    :type n_frames: int
    :return: numpy array of shape (n_frames, n_rois)
    :rtype: np.ndarray
    """
    df = pd.read_csv(filepath, index_col=0)

    if n_frames is not None and df.shape[0] != n_frames:
        raise ValueError(
            f"Expected {n_frames} frames, but got {df.shape[0]} in {filepath}"
        )
    if n_rois is not None and df.shape[1] != n_rois:
        raise ValueError(f"Expected {n_rois} ROIs, but got {df.shape[1]} in {filepath}")

    return df.to_numpy()


def save_NroisxF(
    array: np.ndarray,
    filepath: str,
    header: List[str] = None,
) -> None:
    """
    Saves a Nrois x F array to a CSV file with an index to denote the ROIs and a header for the features

    :param array: Nrois x F array to save
    :type array: np.ndarray
    :param filepath: path of the output file
    :type filepath: str
    :param header: names of the F feature columns, defaults to None (numbered columns)
    :type header: List[str], optional
    """
    if not isinstance(array, np.ndarray):
        raise TypeError("Input must be a numpy ndarray")
    if array.ndim != 2:
        raise ValueError("Input array must be 2D (Nrois x F)")
    if header is not None and len(header) != array.shape[1]:
        raise ValueError("Header length must match the number of columns in the array")
    df = pd.DataFrame(array, columns=header)
    df.index.name = "ROIs"
    _make_parent_dir(filepath)
    df.to_csv(filepath, index=True, compression="zstd")


def concat_analysis_files(root_folder: str, filetype: str = "output") -> None:
    """
    Concatenates all *_analysis_output.csv files in the given root folder and its subdirectories into a single CSV file named experiment_overview.csv.

    :param root_folder: folder where *_analysis_output.csv files are located
    :type root_folder: str
    :raises FileNotFoundError: if no *_analysis_output.csv files are found
    """
    root = Path(root_folder)
    # Search recursively for *_analysis_output.csv
    if filetype == "output":
        csv_files = [
            f
            for f in root.rglob("*_analysis_output.csv")
            if not f.name.startswith(".")  # skip hidden files
        ]
        if not csv_files:
            raise FileNotFoundError(
                "No *_analysis_output.csv files found in the directory or subdirectories."
            )
        output_path = root / "analysis_output_overview.csv"
    elif filetype == "parameters":
        csv_files = [
            f
            for f in root.rglob("*_analysis_parameters.csv")
            if not f.name.startswith(".")  # skip hidden files
        ]
        if not csv_files:
            raise FileNotFoundError(
                "No *_analysis_parameters.csv files found in the directory or subdirectories."
            )
        output_path = root / "analysis_params_overview.csv"
    else:
        raise ValueError(
            f"Unexpected filetype: {filetype}, use 'parameters' or 'output'"
        )
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        # Record the relative path for context
        df["source_file"] = str(f.relative_to(root))
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)

    combined.to_csv(output_path, index=False)
    logger.info(f"Saved combined CSV to {output_path}")
