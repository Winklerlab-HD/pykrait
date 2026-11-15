import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

import pykrait.gui.fileselection as fileselection  # noqa: E402


@pytest.fixture
def window(monkeypatch):
    QApplication.instance() or QApplication([])
    saved = {}
    monkeypatch.setattr(
        fileselection.FileSelectionWindow,
        "load_settings",
        lambda self: {"frame_interval": 2.5},
    )
    monkeypatch.setattr(
        fileselection.FileSelectionWindow,
        "save_settings",
        lambda self: saved.update(self.config),
    )
    monkeypatch.setattr(fileselection, "start_worker", lambda *args, **kwargs: None)
    window = fileselection.FileSelectionWindow()
    window.saved_settings = saved
    yield window
    window.close()


@pytest.mark.parametrize("mode", ["model_radio", "label_radio", "label_csv_radio"])
def test_frame_interval_box_is_shown_and_used_in_every_mode(window, mode):
    getattr(window, mode).setChecked(True)
    assert window.frame_input.isVisibleTo(window)
    assert window.std_button.isVisibleTo(window) == (mode == "model_radio")

    window.image_path = "/data/video.czi"
    window.model_path = "/data/model"
    window.label_image_path = "/data/masks.tif"
    window.mean_intensities_path = "/data/intensities.csv.zst"
    window.frame_input.setText("1.234")
    window.confirm_selection()

    assert window.analysis_params.frame_interval == 1.234
    assert window.saved_settings["frame_interval"] == 1.234


@pytest.mark.parametrize("checked", [True, False])
def test_save_temp_file_checkbox_is_remembered_and_passed_to_the_analysis(
    window, checked
):
    assert window.temp_file_checkbox.isChecked()  # on by default
    window.temp_file_checkbox.setChecked(checked)
    window.image_path = "/data/video.czi"
    window.model_path = "/data/model"
    window.confirm_selection()

    assert window.saved_settings["save_temp_file"] is checked
    assert window.worker.save_temp_file is checked


def test_decimal_validator_accepts_a_comma(window):
    from PySide6.QtCore import QLocale
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit

    from pykrait.gui.custom_widgets import DecimalValidator

    QLocale.setDefault(QLocale(QLocale.German))
    try:
        line_edit = QLineEdit()
        line_edit.setValidator(DecimalValidator(0.0, 1.0, 2))
        QTest.keyClicks(line_edit, "0,25")
        assert line_edit.text() == "0.25" and line_edit.hasAcceptableInput()
        assert float(line_edit.text()) == 0.25
    finally:
        QLocale.setDefault(QLocale.c())


def test_frame_interval_box_accepts_a_comma(window):
    window.image_path = "/data/video.czi"
    window.model_path = "/data/model"
    window.frame_input.setText("2,5")
    window.confirm_selection()
    assert window.analysis_params.frame_interval == 2.5


@pytest.mark.parametrize("typed, expected", [("2.0", 2.0), ("", None)])
def test_typed_frame_interval_wins_over_loaded_parameters_in_every_mode(
    window, typed, expected
):
    from pykrait.pipeline.pipeline import AnalysisParameters

    window.label_radio.setChecked(True)
    window.image_path = "/data/video.czi"
    window.label_image_path = "/data/masks.tif"
    window.analysis_params = AnalysisParameters(frame_interval=9.0)
    window.frame_input.setText(typed)
    window.confirm_selection()
    assert window.analysis_params.frame_interval == expected


def test_selecting_another_video_forgets_the_previous_one(
    window, monkeypatch, tmp_path
):
    from pykrait.pipeline.pipeline import AnalysisParameters

    first = tmp_path / "a" / "first.czi"
    second = tmp_path / "b" / "second.czi"
    for video in (first, second):
        video.parent.mkdir()
        video.write_bytes(b"")
    (first.parent / "Analysis_first").mkdir()
    (first.parent / "Analysis_first" / "first_raw_intensities.csv.zst").write_bytes(b"")

    picked = []
    monkeypatch.setattr(
        fileselection.QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(picked[-1]), ""),
    )
    window.label_csv_radio.setChecked(True)
    window.label_image_path = "/data/hand_picked_masks.tif"
    picked.append(first)
    window.select_image(window.lbl_video_vlc)
    assert window.mean_intensities_path.endswith("first_raw_intensities.csv.zst")
    window.analysis_params = AnalysisParameters(frame_interval=9.0)

    picked.append(second)
    window.select_image(window.lbl_video_vlc)
    assert window.analysis_params is None
    assert window.mean_intensities_path is None  # it was found for the first video
    assert window.label_image_path == "/data/hand_picked_masks.tif"  # picked by hand
