import os
from pathlib import Path
import yaml

from PySide6.QtWidgets import (
    QWidget,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QCheckBox,
    QLineEdit,
    QGroupBox,
    QButtonGroup,
    QRadioButton,
    QProgressBar,
    QStackedWidget,
    QSizePolicy,
    QLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QFormLayout,
)
from PySide6.QtCore import Qt, Signal
from pykrait.pipeline.pipeline import AnalysisParameters, AnalysisOutput
from pykrait.preprocessing.motion_correction import MOTION_CORRECTION_METHODS
from pykrait.gui.async_workers import (
    ExtractIntensitiesWorker,
    defer_close_while_running,
    start_worker,
    stop_workers,
)
from pykrait.io.files import get_analysis_file

GROUP_BOX_STYLE = "QGroupBox { font-weight: bold; font-size: 14px; }"


def _snug_stack() -> QStackedWidget:
    """Returns a stacked widget whose height always fits its current page."""
    stack = QStackedWidget()

    def sync_height():
        height = stack.currentWidget().sizeHint().height()
        stack.setMinimumHeight(height)
        stack.setMaximumHeight(height)
        stack.updateGeometry()

    stack.currentChanged.connect(lambda _: sync_height())
    stack.sync_height = sync_height
    return stack


def _settings_page(title: str):
    """
    Returns a page for a stacked widget holding a single group box, and the layout of
    the group box.
    """
    page = QWidget()
    page_layout = QVBoxLayout(page)
    page_layout.setContentsMargins(0, 0, 0, 0)  # keep the page flush with the group box
    page_layout.setSpacing(0)
    page_layout.setSizeConstraint(QLayout.SetMinimumSize)

    group = QGroupBox(title)
    group.setStyleSheet(GROUP_BOX_STYLE)
    group_layout = QVBoxLayout(group)
    group_layout.setSpacing(6)
    group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
    page_layout.addWidget(group)
    return page, group_layout


class FileSelectionWindow(QWidget):
    analysis_complete = Signal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Microscopy & Model File Selection")
        self.setMinimumSize(500, 550)

        self.config_path = Path.home() / ".pykrait" / "settings.yaml"
        self.config = self.load_settings()

        self.image_path = None
        self.model_path = None
        self.label_image_path = None
        self.mean_intensities_path = None
        # files found in the analysis folder of the selected video, not picked by hand
        self._auto_detected = set()
        # parameters and outputs loaded from a previous analysis of the selected video
        self.analysis_params = None
        self.output_params = None

        self.init_ui()

    def init_ui(self):
        self.main_layout = QVBoxLayout()
        # self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(8)

        # --- Entry Point Selection ---
        entry_group = QGroupBox("Entry Point")
        entry_group.setStyleSheet(GROUP_BOX_STYLE)
        entry_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        entry_layout = QHBoxLayout()
        self.model_radio = QRadioButton("Video + Cellpose Model")
        self.label_radio = QRadioButton("Video + Label Image")
        self.label_csv_radio = QRadioButton(
            "Video + Label Image + Intensities File"
        )  # NEW
        self.model_radio.setChecked(True)
        entry_layout.addWidget(self.model_radio)
        entry_layout.addWidget(self.label_radio)
        entry_layout.addWidget(self.label_csv_radio)
        entry_group.setLayout(entry_layout)
        self.main_layout.addWidget(entry_group)

        # --- File Selection Group ---
        self.file_stack = _snug_stack()
        self.file_stack.addWidget(self.create_video_model_page())  # index 0
        self.file_stack.addWidget(self.create_video_labels_page())  # index 1
        self.file_stack.addWidget(self.create_video_labels_csv_page())  # index 2

        # Connect radio buttons to change stack
        self.model_radio.toggled.connect(lambda: self.file_stack.setCurrentIndex(0))
        self.label_radio.toggled.connect(lambda: self.file_stack.setCurrentIndex(1))
        self.label_csv_radio.toggled.connect(lambda: self.file_stack.setCurrentIndex(2))

        self.file_stack.sync_height()

        self.main_layout.addWidget(self.file_stack)

        # --- Analysis Parameter Status ---
        # greyed out, it only shows the file the parameters were inferred from
        self.load_param_button = QPushButton("Load Analysis Parameters")
        self.load_param_button.setEnabled(False)
        self.main_layout.addWidget(self.load_param_button)

        # --- Analysis Settings Group ---
        self.main_layout.addWidget(self.create_analysis_settings())

        # --- Motion Correction Settings ---
        self.main_layout.addWidget(self.create_motion_correction_group())

        # --- Confirm Button & Progress Bar ---
        confirm_layout = QVBoxLayout()
        confirm_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.confirm_button = QPushButton("Confirm and Proceed")
        self.confirm_button.setStyleSheet(
            "font-weight: bold; padding: 8px 16px; font-size: 14px;"
        )
        self.confirm_button.clicked.connect(self.confirm_selection)
        confirm_layout.addWidget(
            self.confirm_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setVisible(False)
        self.progress_bar.setMinimumWidth(300)
        confirm_layout.addWidget(
            self.progress_bar, alignment=Qt.AlignmentFlag.AlignCenter
        )
        self.progress_label = QLabel(self)
        self.progress_label.setText("")  # Start empty
        confirm_layout.addWidget(
            self.progress_label, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.main_layout.addLayout(confirm_layout)

        self.setLayout(self.main_layout)

    # ---------- SETTINGS LOAD/SAVE ----------
    def load_settings(self):
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def save_settings(self):
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f)

    def get_last_directory(self, key: str):
        return self.config.get(key, str(Path.home()))

    def set_last_directory(self, key: str, path: str):
        self.config[key] = os.path.dirname(path)
        self.save_settings()

    # ---------- FILE SELECTION LAYOUT ----------
    def _file_page(self, rows) -> QWidget:
        """
        Builds a file selection page with a browse button and a label for every file.

        :param rows: tuples of (suffix of the attribute names, button text, file picker)
        :return: returns the page
        :rtype: QWidget
        """
        page, g_layout = _settings_page("File Selection")
        for name, text, picker in rows:
            button = QPushButton(text)
            label = QLabel("No file selected")
            label.setWordWrap(True)
            button.clicked.connect(
                lambda _=False, picker=picker, label=label: picker(label=label)
            )
            setattr(self, f"btn_{name}", button)
            setattr(self, f"lbl_{name}", label)
            g_layout.addWidget(button)
            g_layout.addWidget(label)
        g_layout.addStretch(1)
        return page

    def create_video_model_page(self) -> QWidget:
        return self._file_page(
            [
                ("video_vm", "Browse Video", self.select_image),
                ("model_vm", "Browse Cellpose Model", self.select_model),
            ]
        )

    def create_video_labels_page(self) -> QWidget:
        return self._file_page(
            [
                ("video_vl", "Browse Video", self.select_image),
                ("labels_vl", "Browse Label Image", self.select_label_image),
            ]
        )

    def create_video_labels_csv_page(self) -> QWidget:
        return self._file_page(
            [
                ("video_vlc", "Browse Video", self.select_image),
                ("labels_vlc", "Browse Label Image", self.select_label_image),
                ("csv_vlc", "Browse Raw Intensities", self.select_mean_intensities),
            ]
        )

    # ---------- FILE PICKERS ----------
    def select_image(self, label):
        last_dir = self.get_last_directory("last_image_dir")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Microscopy Image",
            last_dir,
            filter="Image Files (*.czi *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.image_path = path
            self._forget_previous_video()
            self.set_last_directory("last_image_dir", path)
            filename = os.path.basename(path)
            label.setText(f"Selected Image: {filename}")
            label.setToolTip(path)

            # Look for the analysis files of this video. Videos that already lie in an
            # analysis folder use that folder instead of a nested one.

            # try to auto-load analysis parameters
            param_file = Path(get_analysis_file(path, "analysis_parameters.csv"))
            param_output = Path(get_analysis_file(path, "analysis_output.csv"))
            if (
                param_file.exists()
                and param_output.exists()
                and (self.label_csv_radio.isChecked() or self.model_radio.isChecked())
            ):
                try:
                    self.analysis_params = AnalysisParameters.from_csv(str(param_file))
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Load Failed",
                        f"Could not load parameters from {param_file}:\n{e}",
                    )
                try:
                    self.output_params = AnalysisOutput.from_csv(str(param_output))
                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Load Failed",
                        f"Could not load outputs from {param_output}:\n{e}",
                    )
                self.load_param_button.setEnabled(False)
                self.load_param_button.setText(
                    f"Inferred Parameters from {param_file.name}"
                )
            # try to auto-load masks
            mask_file = Path(get_analysis_file(path, "cp_masks.ome.tif"))
            if mask_file.exists() and (
                self.label_radio.isChecked() or self.label_csv_radio.isChecked()
            ):
                self.label_image_path = str(mask_file)
                self._auto_detected.add("label_image_path")
                for label in (self.lbl_labels_vl, self.lbl_labels_vlc):
                    label.setText(f"Auto-detected Mask: {mask_file.name}")
                    label.setToolTip(str(mask_file))

            # try to auto-load raw intensities
            intensities_file = Path(get_analysis_file(path, "raw_intensities.csv.zst"))
            if intensities_file.exists() and self.label_csv_radio.isChecked():
                self.mean_intensities_path = str(intensities_file)
                self._auto_detected.add("mean_intensities_path")
                self.lbl_csv_vlc.setText(
                    f"Auto-detected Intensities: {intensities_file.name}"
                )
                self.lbl_csv_vlc.setToolTip(str(intensities_file))

    def _forget_previous_video(self) -> None:
        """
        Drops everything that belonged to the previously selected video: its loaded
        parameters and outputs, and the masks and intensities found in its analysis
        folder. Files the user picked by hand are kept.
        """
        self.analysis_params = None
        self.output_params = None
        self.load_param_button.setText("Load Analysis Parameters")
        if "label_image_path" in self._auto_detected:
            self.label_image_path = None
            for label in (self.lbl_labels_vl, self.lbl_labels_vlc):
                label.setText("No file selected")
                label.setToolTip("")
        if "mean_intensities_path" in self._auto_detected:
            self.mean_intensities_path = None
            self.lbl_csv_vlc.setText("No file selected")
            self.lbl_csv_vlc.setToolTip("")
        self._auto_detected = set()

    def select_model(self, label):
        last_dir = self.get_last_directory("last_model_dir")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Cellpose Model", last_dir, filter="All Files (*)"
        )
        if path:
            self.model_path = path
            self.set_last_directory("last_model_dir", path)
            filename = os.path.basename(path)
            label.setText(f"Selected Model: {filename}")
            label.setToolTip(path)

    def select_label_image(self, label):
        last_dir = self.get_last_directory("last_image_dir")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Label Image",
            last_dir,
            filter="Label Images (*masks*);;Image Files (*.tif *.tiff *.npy, *.png);;All Files (*)",
        )
        if path:
            self.label_image_path = path
            self._auto_detected.discard("label_image_path")
            filename = os.path.basename(path)
            label.setText(f"Selected Label Image: {filename}")
            label.setToolTip(path)

    def select_mean_intensities(self, label):
        last_dir = self.get_last_directory("last_image_dir")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Raw Intensities",
            last_dir,
            filter="Raw Intensities CSV (*raw_intensities*.zst);;CSV Files (*.csv.zst);;All Files (*)",
        )
        if path:
            self.mean_intensities_path = path
            self._auto_detected.discard("mean_intensities_path")
            filename = os.path.basename(path)
            label.setText(f"Selected Intensity File: {filename}")
            label.setToolTip(path)

    # ---------- ANALYSIS SETTINGS  ----------
    def create_analysis_settings(self) -> QWidget:
        """
        Builds the analysis settings. The projection and CLAHE settings are only used to
        segment with Cellpose and only shown in that mode, the frame interval box is
        shared by all modes.
        """
        page, g_layout = _settings_page("Analysis Settings")
        # the height follows the rows that are shown in the current mode
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        self.cellpose_settings = QWidget()
        cellpose_layout = QVBoxLayout(self.cellpose_settings)
        cellpose_layout.setContentsMargins(0, 0, 0, 0)
        cellpose_layout.setSpacing(6)

        # T-projection method
        tproj_row = QHBoxLayout()
        tproj_label = QLabel("T-projection method (required):")
        tproj_row.addWidget(tproj_label)
        self.tproj_group = QButtonGroup(self)
        self.std_button = QRadioButton("STD")
        self.sum_button = QRadioButton("SUM")
        self.tproj_group.addButton(self.std_button)
        self.tproj_group.addButton(self.sum_button)
        tproj_row.addWidget(self.std_button)
        tproj_row.addWidget(self.sum_button)
        tproj_row.addStretch()
        cellpose_layout.addLayout(tproj_row)

        # Restore saved T-projection
        saved_proj = self.config.get("t_projection", "STD").upper()
        (self.std_button if saved_proj == "STD" else self.sum_button).setChecked(True)

        # CLAHE
        self.clahe_checkbox = QCheckBox("Apply CLAHE normalization")
        self.clahe_checkbox.setChecked(self.config.get("clahe", False))
        cellpose_layout.addWidget(self.clahe_checkbox)

        g_layout.addWidget(self.cellpose_settings)
        self.cellpose_settings.setVisible(self.model_radio.isChecked())
        self.model_radio.toggled.connect(self.cellpose_settings.setVisible)

        # Frame interval
        frame_interval_layout = QHBoxLayout()
        frame_interval_layout.addWidget(QLabel("Frame interval (s, optional):"))
        self.frame_input = QLineEdit()
        self.frame_input.setPlaceholderText("e.g. 2.50")
        if "frame_interval" in self.config:
            self.frame_input.setText(str(self.config["frame_interval"]))
        frame_interval_layout.addWidget(self.frame_input)
        g_layout.addLayout(frame_interval_layout)

        # the video is read once if its intensity channel fits into memory; otherwise
        # it can be kept in a temporary file instead of reading the video again
        self.temp_file_checkbox = QCheckBox("Save temporary file on disk")
        self.temp_file_checkbox.setToolTip(
            "If the intensity channel of the video does not fit into memory, keep it in a "
            "temporary file on the internal drive instead of reading the video a second "
            "time. Faster for videos on external or network drives."
        )
        self.temp_file_checkbox.setChecked(self.config.get("save_temp_file", True))
        g_layout.addWidget(self.temp_file_checkbox)

        g_layout.addStretch(1)
        return page

    # ---------- MOTION CORRECTION SETTINGS ----------
    def create_motion_correction_group(self) -> QGroupBox:
        """
        Builds the motion correction settings. The correction is disabled by default and
        its settings are only enabled once the user ticks the checkbox.

        :return: returns the group box holding the motion correction settings
        :rtype: QGroupBox
        """
        group = QGroupBox("Motion Correction")
        group.setStyleSheet(GROUP_BOX_STYLE)
        group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        g_layout = QVBoxLayout(group)
        g_layout.setSpacing(6)

        self.motion_correction_checkbox = QCheckBox("Correct lateral (xy) motion")
        self.motion_correction_checkbox.setToolTip(
            "Aligns every frame to a template before segmentation and intensity extraction. "
            "The corrected video is saved into the analysis folder."
        )
        self.motion_correction_checkbox.setChecked(
            bool(self.config.get("motion_correction", False))
        )
        g_layout.addWidget(self.motion_correction_checkbox)

        # container for all settings, greyed out while the correction is disabled
        self.motion_settings_widget = QWidget()
        settings_layout = QFormLayout(self.motion_settings_widget)
        settings_layout.setContentsMargins(0, 0, 0, 0)
        settings_layout.setSpacing(5)

        self.motion_method_combo = QComboBox()
        self.motion_method_combo.addItems(MOTION_CORRECTION_METHODS)
        saved_method = self.config.get("motion_correction_method", "warp_grid")
        if saved_method in MOTION_CORRECTION_METHODS:
            self.motion_method_combo.setCurrentText(saved_method)
        settings_layout.addRow("Method:", self.motion_method_combo)

        self.motion_channel_spin = QSpinBox()
        self.motion_channel_spin.setMinimum(0)
        self.motion_channel_spin.setMaximum(9)
        self.motion_channel_spin.setValue(
            int(self.config.get("motion_correction_channel", 1))
        )
        self.motion_channel_spin.setToolTip(
            "Channel the motion is estimated on. Works best on a channel with stationary labeling. "
            "Single-channel videos always use channel 0."
        )
        settings_layout.addRow("Estimate on channel:", self.motion_channel_spin)

        self.motion_template_frames_spin = QSpinBox()
        self.motion_template_frames_spin.setMinimum(1)
        self.motion_template_frames_spin.setMaximum(1000)
        self.motion_template_frames_spin.setValue(
            int(self.config.get("motion_correction_template_frames", 10))
        )
        settings_layout.addRow("Template frames:", self.motion_template_frames_spin)

        self.motion_average_frames_spin = QSpinBox()
        self.motion_average_frames_spin.setMinimum(1)
        self.motion_average_frames_spin.setMaximum(99)
        self.motion_average_frames_spin.setValue(
            int(self.config.get("motion_correction_average_frames", 3))
        )
        self.motion_average_frames_spin.setToolTip(
            "Number of frames averaged before the motion is estimated. The correction "
            "itself is always applied to the single frame."
        )
        settings_layout.addRow("Averaged frames:", self.motion_average_frames_spin)

        self.motion_template_filter_spin = QDoubleSpinBox()
        self.motion_template_filter_spin.setDecimals(3)
        self.motion_template_filter_spin.setSingleStep(0.01)
        self.motion_template_filter_spin.setMinimum(0.0)
        self.motion_template_filter_spin.setMaximum(1.0)
        self.motion_template_filter_spin.setValue(
            float(self.config.get("motion_correction_template_filter", 0.0))
        )
        self.motion_template_filter_spin.setToolTip(
            "Factor to update the template with every corrected frame. 0 keeps the template fixed."
        )
        settings_layout.addRow("Template filter:", self.motion_template_filter_spin)

        self.motion_grid_size_spin = QSpinBox()
        self.motion_grid_size_spin.setMinimum(2)
        self.motion_grid_size_spin.setMaximum(100)
        self.motion_grid_size_spin.setValue(
            int(self.config.get("motion_correction_grid_size", 10))
        )
        self.motion_grid_size_label = QLabel("Mesh grid size:")
        settings_layout.addRow(self.motion_grid_size_label, self.motion_grid_size_spin)

        self.motion_polynomial_degree_spin = QSpinBox()
        self.motion_polynomial_degree_spin.setMinimum(1)
        self.motion_polynomial_degree_spin.setMaximum(5)
        self.motion_polynomial_degree_spin.setValue(
            int(self.config.get("motion_correction_polynomial_degree", 2))
        )
        self.motion_polynomial_degree_label = QLabel("Polynomial degree:")
        settings_layout.addRow(
            self.motion_polynomial_degree_label, self.motion_polynomial_degree_spin
        )

        self.motion_mask_fov_checkbox = QCheckBox(
            "Mask pixels that left the field of view"
        )
        self.motion_mask_fov_checkbox.setChecked(
            bool(self.config.get("motion_correction_mask_fov", False))
        )
        settings_layout.addRow(self.motion_mask_fov_checkbox)

        g_layout.addWidget(self.motion_settings_widget)

        self.motion_correction_checkbox.toggled.connect(
            self.motion_settings_widget.setEnabled
        )
        self.motion_method_combo.currentTextChanged.connect(
            self._update_motion_method_fields
        )
        self.motion_settings_widget.setEnabled(
            self.motion_correction_checkbox.isChecked()
        )
        self._update_motion_method_fields(self.motion_method_combo.currentText())

        return group

    def _update_motion_method_fields(self, method: str) -> None:
        """
        Only shows the settings that are used by the currently selected method.

        :param method: currently selected motion correction method
        :type method: str
        """
        is_grid = method == "warp_grid"
        is_polynomial = method == "warp_polynomial_fit"

        self.motion_grid_size_label.setVisible(is_grid)
        self.motion_grid_size_spin.setVisible(is_grid)
        self.motion_polynomial_degree_label.setVisible(is_polynomial)
        self.motion_polynomial_degree_spin.setVisible(is_polynomial)

    def apply_motion_correction_settings(self, params: AnalysisParameters) -> None:
        """
        Writes the motion correction settings into the analysis parameters and stores
        them in the user settings.

        :param params: analysis parameters to be updated
        :type params: AnalysisParameters
        """
        settings = {
            "motion_correction": self.motion_correction_checkbox.isChecked(),
            "motion_correction_method": self.motion_method_combo.currentText(),
            "motion_correction_channel": self.motion_channel_spin.value(),
            "motion_correction_template_frames": self.motion_template_frames_spin.value(),
            "motion_correction_average_frames": self.motion_average_frames_spin.value(),
            "motion_correction_template_filter": self.motion_template_filter_spin.value(),
            "motion_correction_grid_size": self.motion_grid_size_spin.value(),
            "motion_correction_polynomial_degree": self.motion_polynomial_degree_spin.value(),
            "motion_correction_mask_fov": self.motion_mask_fov_checkbox.isChecked(),
        }
        for name, value in settings.items():
            setattr(params, name, value)
            self.config[name] = value
        self.save_settings()

    # ---------- CONFIRM ----------
    def confirm_selection(self):
        if not self.image_path:
            QMessageBox.warning(
                self, "Missing File", "Please select a microscopy image."
            )
            return

        if self.model_radio.isChecked():
            if not self.model_path:
                QMessageBox.warning(
                    self, "Missing File", "Please select a Cellpose model."
                )
                return
        elif self.label_csv_radio.isChecked() and not self.mean_intensities_path:
            QMessageBox.warning(
                self, "Missing File", "Please select a mean intensities CSV file."
            )
            return
        else:
            if not self.label_image_path:
                QMessageBox.warning(
                    self, "Missing File", "Please select a label image."
                )
                return

        if not (self.std_button.isChecked() or self.sum_button.isChecked()):
            QMessageBox.warning(
                self, "Missing Selection", "Please select a T-projection method."
            )
            return

        tproj = "std" if self.std_button.isChecked() else "sum"
        clahe = self.clahe_checkbox.isChecked()
        frame_text = self.frame_input.text().strip()
        frame_interval = None
        if frame_text:
            try:
                frame_interval = float(frame_text.replace(",", "."))
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    "Frame interval must be a number (e.g. 2.50).",
                )
                return
            # remembered as entered, a rounded value would be used by the next analysis
            self.config["frame_interval"] = frame_interval
        else:
            self.config.pop("frame_interval", None)

        self.config["t_projection"] = tproj.upper()
        self.config["clahe"] = clahe
        save_temp_file = self.temp_file_checkbox.isChecked()
        self.config["save_temp_file"] = save_temp_file
        self.save_settings()

        # Start progress bar
        self.progress_bar.setVisible(True)
        self.repaint()

        # --- Run Analysis ---
        cellpose_model_path = self.model_path if self.model_radio.isChecked() else None

        # TODO - allow overriding analysis/output params here
        if self.analysis_params is None:
            self.analysis_params = AnalysisParameters(
                tproj_type=tproj,
                CLAHE_normalize=clahe,
                cellpose_model_path=cellpose_model_path,
                frame_interval=frame_interval,
            )
        else:
            if self.model_radio.isChecked():
                self.analysis_params.tproj_type = tproj
                self.analysis_params.CLAHE_normalize = clahe
                self.analysis_params.cellpose_model_path = cellpose_model_path
            # the frame interval box is shown in every mode and always wins over loaded
            # parameters; left empty, the frame interval of the metadata is used
            self.analysis_params.frame_interval = frame_interval

        # the motion correction widgets always win over loaded parameters
        self.apply_motion_correction_settings(self.analysis_params)

        if self.output_params is None:
            self.output_params = AnalysisOutput(filepath=self.image_path)
            self.output_params.filename = os.path.basename(self.output_params.filepath)
            self.output_params.masks_path = (
                self.label_image_path if not self.model_radio.isChecked() else None
            )
        elif self.label_radio.isChecked() or self.label_csv_radio.isChecked():
            self.output_params.masks_path = self.label_image_path

        if self.model_radio.isChecked():
            mode = "cellpose"
        elif self.label_radio.isChecked():
            mode = "label_image"
        elif self.label_csv_radio.isChecked():
            mode = "csv"
        else:
            QMessageBox.warning(
                self, "Selection Error", "Please select a valid entry point."
            )
            return
        self.worker = ExtractIntensitiesWorker(
            self.analysis_params,
            self.output_params,
            mode,
            self.mean_intensities_path,
            save_temp_file=save_temp_file,
        )
        self.worker.progress_changed.connect(self.update_progress)

        # Show progress bar
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start analysis thread, blocking repeated starts until it is done
        self.confirm_button.setEnabled(False)
        start_worker(self, self.worker, self.analysis_done, self.analysis_failed)

    # ---------- ANALYSIS WORKER -----------
    def update_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.progress_label.setText(message)

    def analysis_done(self, results):
        self.progress_bar.setVisible(False)
        if getattr(self, "close_when_finished", False):
            return  # the user closed the window, so no main window is opened
        self.analysis_complete.emit(results)

    def analysis_failed(self, error_msg):
        self.progress_bar.setVisible(False)
        self.confirm_button.setEnabled(True)
        if getattr(self, "close_when_finished", False):
            return
        QMessageBox.critical(self, "Analysis Error", error_msg)

    def closeEvent(self, event):
        if defer_close_while_running(self, event):
            return
        # a running QThread that is destroyed with its window aborts the application
        stop_workers(self)
        super().closeEvent(event)
