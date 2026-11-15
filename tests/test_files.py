import os

import numpy as np
import pandas as pd
import pytest

from pykrait.io.files import save_NroisxF, save_Txnrois


def test_save_Txnrois_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    array = np.arange(6, dtype=float).reshape(3, 2)
    save_Txnrois(array, 0.5, "traces.csv.zst")

    assert os.path.exists(tmp_path / "traces.csv.zst")
    df = pd.read_csv(tmp_path / "traces.csv.zst", index_col=0)
    np.testing.assert_allclose(df.values, array)


def test_save_Txnrois_small_frame_interval_keeps_unique_index(tmp_path):
    filepath = tmp_path / "fast.csv.zst"
    save_Txnrois(np.zeros((10, 1)), 0.002, str(filepath))

    df = pd.read_csv(filepath, index_col=0)
    assert df.index.is_unique
    np.testing.assert_allclose(df.index.to_numpy(), np.arange(10) * 0.002)


def test_save_Txnrois_creates_parent_folder(tmp_path):
    filepath = tmp_path / "new_folder" / "traces.csv.zst"
    save_Txnrois(np.zeros((3, 2)), 1.0, str(filepath))
    assert filepath.exists()


def test_save_NroisxF_bare_filename(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    array = np.array([[1.0, 0.1], [2.0, 0.2]])
    save_NroisxF(array, "stds.csv.zst", header=["STD", "CoV"])

    df = pd.read_csv(tmp_path / "stds.csv.zst", index_col=0)
    assert list(df.columns) == ["STD", "CoV"]
    np.testing.assert_allclose(df.values, array)


def test_save_NroisxF_without_header(tmp_path):
    filepath = tmp_path / "features.csv.zst"
    array = np.array([[1.0, 0.1, 5.0], [2.0, 0.2, 6.0]])
    save_NroisxF(array, str(filepath))

    df = pd.read_csv(filepath, index_col=0)
    assert df.shape == (2, 3)
    np.testing.assert_allclose(df.values, array)


def test_save_NroisxF_header_length_mismatch_raises(tmp_path):
    with pytest.raises(ValueError, match="Header length"):
        save_NroisxF(np.zeros((2, 2)), str(tmp_path / "x.csv.zst"), header=["only_one"])


def test_get_analysis_file_uses_the_analysis_folder_and_stem(tmp_path):
    from pykrait.io.files import get_analysis_file

    video = tmp_path / "video.ome.tif"
    assert get_analysis_file(str(video), "cp_masks.ome.tif") == str(
        tmp_path / "Analysis_video" / "video_cp_masks.ome.tif"
    )
    assert get_analysis_file(str(video), "peaks.csv.zst", "elsewhere") == os.path.join(
        "elsewhere", "video_peaks.csv.zst"
    )


def test_progress_messages_go_to_stdout_and_can_be_silenced(capsys):
    import logging

    from pykrait.io.files import concat_analysis_files  # noqa: F401, imports the logger

    logger = logging.getLogger("pykrait.io.files")
    logger.info("progress message")
    assert capsys.readouterr().out == "progress message\n"

    package_logger = logging.getLogger("pykrait")
    level = package_logger.level
    package_logger.setLevel(logging.WARNING)
    try:
        logger.info("hidden")
        assert capsys.readouterr().out == ""
    finally:
        package_logger.setLevel(level)


def test_level_set_before_importing_pykrait_is_kept():
    import subprocess
    import sys

    code = (
        "import logging; logging.getLogger('pykrait').setLevel(logging.WARNING); "
        "import pykrait; print(logging.getLogger('pykrait').level)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.stdout.strip() == str(30)


def test_get_files_from_folder_searches_folders_inside_an_analysis_folder(tmp_path):
    from pykrait.io.files import get_files_from_folder

    root = tmp_path / "Analysis_2026" / "run1"
    (root / "Analysis_v").mkdir(parents=True)
    (root / "v.czi").write_bytes(b"")
    (root / "Analysis_v" / "old.czi").write_bytes(b"")

    # only analysis folders below the searched folder are skipped
    assert get_files_from_folder(str(root), recursive=True) == [str(root / "v.czi")]
