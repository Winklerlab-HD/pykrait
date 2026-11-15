import logging
import numpy as np
import traceback
import dask.array as da
from matplotlib import colormaps
from pathlib import Path
from typing import List

from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QMessageBox,
    QLabel,
    QFormLayout,
    QPushButton,
    QSplitter,
    QSlider,
    QDoubleSpinBox,
    QGroupBox,
    QCheckBox,
    QRadioButton,
    QButtonGroup,
    QLineEdit,
    QScrollArea,
)
from PySide6.QtCore import Qt, QPointF, Signal
from PySide6.QtGui import (
    QImage,
    QPixmap,
    QPen,
    QPolygonF,
    QAction,
    QBrush,
    QTransform,
)
from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QGraphicsPolygonItem,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from pykrait.io.files import get_pykrait_version, concat_analysis_files
from pykrait.io.images import read_projection
from pykrait.pipeline.pipeline import (
    AnalysisParameters,
    AnalysisOutput,
    neighbour_tolerance_in_pixels,
    peak_widths_in_frames,
    do_peak_analysis,
    calculate_periodic_cells,
    create_roi_dataframe,
    calculate_synchronicity,
    save_analysis_results,
)
from pykrait.preprocessing.timeseries_extraction import get_adjacency_matrix
from pykrait.trace_analysis.filtering import detrend_with_sinc_filter
from pykrait.gui.async_workers import (
    FunctionWorker,
    defer_close_while_running,
    start_worker,
    stop_workers,
    workers_running,
)
from pykrait.gui.custom_widgets import ToggleSwitch, CollapsibleBox, DecimalValidator

logger = logging.getLogger(__name__)


def _find_periodic_cells(peaks, analysis_params, analysis_output) -> tuple:
    """Finds the periodic ROIs and collects the per-ROI results, run on a worker thread."""
    analysis_output, stds, covs = calculate_periodic_cells(
        peaks, analysis_params, analysis_output
    )
    roi_dataframe = create_roi_dataframe(peaks, stds, covs, analysis_output)
    return analysis_output, stds, covs, roi_dataframe


# decimals of the frame interval box, so that e.g. 30 fps (0.0333 s) is not shown as 0.03
FRAME_INTERVAL_DECIMALS = 4


def _double_spin_box(
    decimals: int, minimum: float, maximum: float, value: float
) -> QDoubleSpinBox:
    """Creates a QDoubleSpinBox, the value is set after the range it is clamped to."""
    spin_box = QDoubleSpinBox()
    spin_box.setDecimals(decimals)
    spin_box.setMinimum(minimum)
    spin_box.setMaximum(maximum)
    spin_box.setValue(value)
    return spin_box


def _save_with_projection(tproj_path: str, **kwargs) -> None:
    """Reads the projection and saves the analysis results, run on a worker thread."""
    save_analysis_results(tproj=read_projection(tproj_path), **kwargs)


class ImageDisplay(QGraphicsView):
    roiSelected = Signal(int)

    def __init__(self, parent=None, max_display_size=512):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item = QGraphicsPixmapItem()
        self.scene.addItem(self.image_item)
        self.roi_items = []
        self.rois = []
        # frames are downsampled for display, but the scene is kept in original pixel
        # coordinates by scaling the pixmap back up, so ROIs never need rescaling
        self.max_display_size = max_display_size
        self.display_factor = 1
        self.selected_roi = None  # to track the active ROI
        self.intensity_clip_min = None
        self.intensity_clip_max = None

    def set_rois(self, rois):
        # Clear old ROIs
        for item in self.roi_items:
            self.scene.removeItem(item)
        self.roi_items.clear()
        self.rois = rois

        for i, poly_pts in enumerate(rois):
            polygon = QPolygonF([QPointF(x, y) for x, y in poly_pts])
            pen = QPen(Qt.white)  # White color
            pen.setWidth(1)  # Line width
            pen.setStyle(Qt.DashLine)  # Dashed line
            pen.setCosmetic(True)  # width in screen pixels, independent of zoom
            roi_item = QGraphicsPolygonItem(polygon)
            roi_item.setPen(pen)
            roi_item.setBrush(Qt.NoBrush)  # No fill
            roi_item.setZValue(1)
            roi_item.setData(0, i)
            self.scene.addItem(roi_item)
            self.roi_items.append(roi_item)
        self.highlight_selected_roi()

    def highlight_selected_roi(self):
        for i, roi_item in enumerate(self.roi_items):
            if i == self.selected_roi:
                pen = QPen(Qt.red)
                pen.setWidth(3)
                pen.setStyle(Qt.SolidLine)
                roi_item.setZValue(100)  # Highest level for selected
            else:
                pen = QPen(Qt.white)
                pen.setWidth(1)
                pen.setStyle(Qt.DashLine)
                roi_item.setZValue(1)  # Normal level for unselected
            pen.setCosmetic(True)
            roi_item.setPen(pen)
            roi_item.setBrush(Qt.NoBrush)

    def draw_roi_centers(self, indices: List[int], radius: int = 5) -> None:
        """
        draws red dots at the centers of the specified ROIs.

        :param indices: list of indices of ROIs to mark
        :type indices: List[int]
        :param radius: radius of the highlight, defaults to 3
        :type radius: int, optional
        """
        # Clear old center markers first (if needed)
        self.clear_roi_centers()

        self.center_items = []
        # the radius is given in displayed pixels, the scene is in original pixels
        radius = radius * self.display_factor

        for i in indices:
            if i < 0 or i >= len(self.rois):
                continue  # Skip invalid indices

            poly_pts = self.rois[i]
            if not poly_pts:
                continue

            # Compute centroid
            xs, ys = zip(*poly_pts)
            cx = sum(xs) / len(xs)
            cy = sum(ys) / len(ys)

            # Create red dot (ellipse)
            dot = self.scene.addEllipse(
                cx - radius,
                cy - radius,
                2 * radius,
                2 * radius,
                QPen(Qt.green),
                QBrush(Qt.green),
            )
            dot.setZValue(2)  # Above polygons
            self.center_items.append(dot)

    def clear_roi_centers(self):
        if hasattr(self, "center_items"):
            for item in self.center_items:
                self.scene.removeItem(item)
            self.center_items.clear()

    def draw_roi_connections(self, shortest_path_matrix) -> None:
        """
        Draw lines between ROI centers for all connected pairs
        according to shortest_path_matrix.
        """
        # Clear old lines
        self.clear_roi_connections()

        # Compute all ROI centroids
        centers = {}
        for i, poly_pts in enumerate(self.rois):
            if not poly_pts:
                continue
            xs, ys = zip(*poly_pts)
            centers[i] = (sum(xs) / len(xs), sum(ys) / len(ys))

        self.connection_items = []
        n = len(self.rois)
        for i in range(len(self.rois)):
            for j in range(i + 1, n):  # only upper triangle, avoid self-loops
                dist = shortest_path_matrix[i][j]
                if dist == 1:
                    if i in centers and j in centers:
                        x1, y1 = centers[i]
                        x2, y2 = centers[j]
                        pen = QPen(Qt.blue, 2)
                        pen.setCosmetic(True)
                        line = self.scene.addLine(x1, y1, x2, y2, pen)
                        line.setZValue(1)  # behind the dots
                        self.connection_items.append(line)

    def clear_roi_connections(self):
        if hasattr(self, "connection_items"):
            for item in self.connection_items:
                self.scene.removeItem(item)
            self.connection_items.clear()

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        for idx, roi_item in enumerate(self.roi_items):
            if roi_item.contains(scene_pos):
                self.selected_roi = idx
                self.highlight_selected_roi()
                self.roiSelected.emit(idx)  # <---- emit the signal
                break
        else:
            self.selected_roi = None
            self.highlight_selected_roi()
        super().mousePressEvent(event)

    def display_dask_frame(self, frame: da.Array, cmap="magma"):
        """
        displays a single frame from a dask array in the QGraphicsView.
        Uses coarsening to downsample the frame, and normalizes the brightness consistently.

        :param frame: dask array representing a single frame of the timelapse
        :type frame: da.Array
        :param cmap: colormap to display in, defaults to 'magma'
        :type cmap: str, optional
        :raises ValueError: raises ValueError if the input frame is not a 2D array
        """
        if frame.ndim != 2:
            raise ValueError("Input frame must be a 2D array")
        h, w = frame.shape

        # the same factor on both axes keeps the aspect ratio
        factor = max(1, max(h, w) // self.max_display_size)
        self.display_factor = factor

        # Downsample using mean pooling
        frame_ds = da.coarsen(np.mean, frame, {0: factor, 1: factor}, trim_excess=True)
        img = frame_ds.compute()

        # Normalize the image for optimized display
        if self.intensity_clip_min is None:
            self.intensity_clip_min = np.percentile(
                img, 1
            )  # 1st percentile for better contrast
        if self.intensity_clip_max is None:
            self.intensity_clip_max = np.percentile(img, 99)
        # Avoid divide-by-zero
        if self.intensity_clip_max - self.intensity_clip_min < 1e-6:
            norm = np.zeros_like(img, dtype=float)
        else:
            norm = np.clip(
                (img - self.intensity_clip_min)
                / (self.intensity_clip_max - self.intensity_clip_min),
                0,
                1,
            )

        # Apply colormap
        img_rgb = (colormaps[cmap](norm)[..., :3] * 255).astype(np.uint8)

        # Convert to QImage and display
        h_ds, w_ds, _ = img_rgb.shape
        qimg = QImage(img_rgb.data, w_ds, h_ds, 3 * w_ds, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.image_item.setPixmap(pix)
        # each displayed pixel covers factor x factor original pixels
        self.image_item.setTransform(QTransform.fromScale(factor, factor))
        self.fitInView(self.image_item, Qt.KeepAspectRatio)


class MainWindow(QMainWindow):
    def __init__(
        self,
        frames: da.Array,
        mask: np.ndarray,
        mean_intensities: np.ndarray,
        rois: list,
        analysis_output: AnalysisOutput,
        analysis_params: AnalysisParameters,
        app_controller=None,
        display_channel: int = 0,
    ):
        super().__init__()
        self.app_controller = app_controller
        self.frames = frames
        self.display_channel = display_channel  # channel the intensities come from
        self.masks = mask
        self.rois = rois
        self.intensities = mean_intensities
        self.analysis_output = analysis_output
        self.analysis_params = analysis_params
        self.setWindowTitle(
            f"{self.analysis_output.filename}; pyKrait {get_pykrait_version()}"
        )

        self.selected_roi = 0  # Default to first ROI
        self.current_frame = 0
        self.show_detrended: bool = False
        self.detrended_intensities = None
        self._detrending = False
        self.peaks = None
        self.last_sinc_window = (
            None  # sinc window the detrended traces were computed with
        )
        self.experimental_stds = None
        self.experimental_covs = None
        self.roi_dataframe = None
        self.adjacency_matrix = None
        self.shortest_path_matrix = None
        self.true_synchronous_peaks_matrix = None
        # menu bar
        self._create_menu_bar()
        # Central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # -------- LEFT PANEL --------
        left_panel_widget = QWidget()
        self.left_panel_layout = QVBoxLayout(left_panel_widget)
        self._build_left_panel()
        self.left_panel_layout.addWidget(self.image_display)
        self.left_panel_layout.addWidget(self.frame_slider)
        self.left_panel_layout.addWidget(self.canvas)

        # -------- RIGHT PANEL --------
        self.right_panel_layout = QVBoxLayout()
        right_panel_widget_content = QWidget()
        right_panel_widget_content.setLayout(self.right_panel_layout)
        self.right_panel_layout.addWidget(self._build_trace_group())
        self.right_panel_layout.addWidget(self._build_peak_group())
        self.right_panel_layout.addWidget(self._build_periodicity_group())
        self.right_panel_layout.addWidget(self._build_synchronicity_group())
        self.right_panel_layout.addStretch()
        self.right_panel_layout.addWidget(self._build_save_group())

        # --- SCROLL AREA SETUP ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)  # allows vertical resizing
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setWidget(right_panel_widget_content)

        # fix the width of the right panel to its content
        width = right_panel_widget_content.sizeHint().width()
        right_panel_widget_content.setMinimumWidth(width)
        right_panel_widget_content.setMaximumWidth(width)
        scroll_area.updateGeometry()

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel_widget)
        splitter.addWidget(scroll_area)

        # Set initial splitter sizes
        initial_right_width = right_panel_widget_content.sizeHint().width()
        splitter.setSizes([700, initial_right_width])

        # Add splitter to main layout
        main_layout.addWidget(splitter)

        # set data
        self.image_display.display_dask_frame(
            self.frames[0, self.display_channel, :, :]
        )  # TCYX format
        self.image_display.set_rois(self.rois)
        self.update_intensity_trace()

        # Connecting buttons and signals
        self.frame_slider.valueChanged.connect(self.on_slider_moved)
        self.show_raw_btn.clicked.connect(self.show_raw_traces)
        self.show_detrended_btn.clicked.connect(self.show_detrended_traces)
        self.image_display.roiSelected.connect(self.roi_selected)
        self.find_peaks_btn.clicked.connect(self._start_peak_worker)
        self.periodicity_button.clicked.connect(self._start_periodicity_worker)
        self.periodic_display_checkbox.stateChanged.connect(
            self._periodic_cell_highlight
        )
        self.periodicity_toggle.toggled.connect(self._periodic_cell_highlight)
        self.show_neighbour_btn.clicked.connect(self._start_neighbours_worker)
        self.calculate_synchronous_btn.clicked.connect(self._start_synchronicity_worker)

    def _build_left_panel(self) -> None:
        """Creates the frame display, the frame slider and the intensity plot."""
        # Image display
        self.image_display = ImageDisplay()
        self.image_display.setMinimumSize(400, 400)

        # Video slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.frames.shape[0] - 1)  # Use -1 for index

        # Intensity plot
        self.fig = Figure(figsize=(4, 2))
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

    def _build_trace_group(self) -> CollapsibleBox:
        trace_group = CollapsibleBox("Trace Detrending")

        # Container widget for the form layout
        form_container = QWidget()
        trace_layout = QFormLayout(form_container)
        trace_layout.setContentsMargins(0, 0, 0, 0)
        trace_layout.setSpacing(5)

        # Sinc filter window
        sinc_max = self.frames.shape[0] * self.analysis_output.frame_interval
        self.sinc_filter_spin = _double_spin_box(
            decimals=2, minimum=1, maximum=sinc_max, value=int(sinc_max / 2)
        )
        trace_layout.addRow("Sinc Filter Window [s]:", self.sinc_filter_spin)

        # Frame interval auto-detected
        self.frame_interval_value = QLabel(f"{self.analysis_output.frame_interval:.4g}")
        trace_layout.addRow(
            "Frame Interval Auto-Detected [s]:", self.frame_interval_value
        )

        # Frame interval spin
        # the box shows the frame interval rounded, see _typed_frame_interval
        self.frame_interval_spin = _double_spin_box(
            decimals=FRAME_INTERVAL_DECIMALS,
            minimum=10**-FRAME_INTERVAL_DECIMALS,
            maximum=1000,
            value=self.analysis_output.frame_interval,
        )
        trace_layout.addRow("Frame Interval [s]:", self.frame_interval_spin)

        # Video duration
        self.video_duration_value = QLabel(
            f"{self.analysis_output.frame_interval * self.intensities.shape[0]:.2f}"
        )
        trace_layout.addRow("Video Duration [s]:", self.video_duration_value)

        # Buttons in a horizontal layout
        trace_btn_layout = QHBoxLayout()
        self.show_detrended_btn = QPushButton("Show Detrended Traces")
        self.show_raw_btn = QPushButton("Show Raw Traces")
        trace_btn_layout.addWidget(self.show_detrended_btn)
        trace_btn_layout.addWidget(self.show_raw_btn)

        btn_container = QWidget()
        btn_container.setLayout(trace_btn_layout)
        trace_layout.addRow(btn_container)

        trace_group.add_widget(form_container)
        return trace_group

    def _build_peak_group(self) -> CollapsibleBox:
        peak_group = CollapsibleBox("Peak Identification")

        # Container widget for the QFormLayout
        form_container = QWidget()
        peak_layout = QFormLayout(form_container)
        peak_layout.setContentsMargins(0, 0, 0, 0)
        peak_layout.setSpacing(5)

        # Peak Min Width
        self.peak_min_width_spin = _double_spin_box(
            decimals=2, minimum=1, maximum=1000, value=1.0
        )
        peak_layout.addRow("Peak Min Width [s]:", self.peak_min_width_spin)

        # Peak Max Width
        self.peak_max_width_spin = _double_spin_box(
            decimals=2, minimum=0, maximum=1000, value=80.0
        )
        peak_layout.addRow("Peak Max Width [s]:", self.peak_max_width_spin)

        # Peak Min Height
        self.peak_min_height_spin = _double_spin_box(
            decimals=0, minimum=0, maximum=65535, value=800
        )
        peak_layout.addRow("Peak Min Height [au]:", self.peak_min_height_spin)

        # Peak Prominence
        self.peak_prominence_spin = _double_spin_box(
            decimals=0, minimum=0, maximum=65535, value=1500
        )
        peak_layout.addRow("Peak Prominence [au]:", self.peak_prominence_spin)

        # Find Peaks Button
        self.find_peaks_btn = QPushButton("Find Peaks")
        peak_layout.addRow(self.find_peaks_btn)

        # Labels
        self.norm_peak_label = QLabel("")
        self.fourpeak_label = QLabel("")
        self.frequency_label = QLabel("")
        peak_layout.addRow("normalized peaks:", self.norm_peak_label)
        peak_layout.addRow("% of cells w/ >= 4 peaks:", self.fourpeak_label)
        peak_layout.addRow("median Frequency mHz:", self.frequency_label)

        # Add the form container to the collapsible box
        peak_group.add_widget(form_container)
        return peak_group

    def _build_periodicity_group(self) -> CollapsibleBox:
        periodicity_group = CollapsibleBox("Periodic Cells")

        # Container widget for the QFormLayout
        form_container = QWidget()
        periodicity_layout = QFormLayout(form_container)
        periodicity_layout.setContentsMargins(0, 0, 0, 0)
        periodicity_layout.setSpacing(5)

        # Row 0: mutually exclusive method buttons
        self.rb_periodicity_cutoff = QRadioButton("Use cut-off")
        self.rb_periodicity_quantile = QRadioButton("Use quantile")
        self.rb_periodicity_cutoff.setChecked(True)

        self.periodicity_method_group = QButtonGroup(self)
        self.periodicity_method_group.addButton(self.rb_periodicity_cutoff)
        self.periodicity_method_group.addButton(self.rb_periodicity_quantile)

        _method_row = QWidget()
        _method_row_layout = QHBoxLayout(_method_row)
        _method_row_layout.setContentsMargins(0, 0, 0, 0)
        _method_row_layout.addWidget(self.rb_periodicity_cutoff)
        _method_row_layout.addWidget(self.rb_periodicity_quantile)
        _method_row_layout.addStretch(1)
        periodicity_layout.addRow(_method_row)

        # Row 1: STD threshold input + labels
        self.periodicity_threshold_std_LineEdit = QLineEdit()
        self.periodicity_threshold_std_LineEdit.setMinimumWidth(40)
        self.periodicity_threshold_std_LineEdit.setValidator(
            DecimalValidator(0.0, 1000.0, 2, self)
        )
        self.periodicity_threshold_std_LineEdit.setText("15.0")

        self.std_cutoff_value_label = QLabel("cutoff: —s")
        self.std_quantile_value_label = QLabel("quantile: —%")

        _std_row = QWidget()
        _std_row_layout = QHBoxLayout(_std_row)
        _std_row_layout.setContentsMargins(0, 0, 0, 0)
        _std_row_layout.addWidget(self.periodicity_threshold_std_LineEdit)
        _std_row_layout.addSpacing(12)
        _std_row_layout.addWidget(self.std_cutoff_value_label)
        _std_row_layout.addSpacing(12)
        _std_row_layout.addWidget(self.std_quantile_value_label)
        _std_row_layout.addStretch(1)
        periodicity_layout.addRow("STD threshold:", _std_row)

        # Row 2: CoV threshold input + labels
        self.periodicity_threshold_cov_LineEdit = QLineEdit()
        self.periodicity_threshold_cov_LineEdit.setMinimumWidth(40)
        self.periodicity_threshold_cov_LineEdit.setValidator(
            DecimalValidator(0.0, 1.0, 2, self)
        )
        self.periodicity_threshold_cov_LineEdit.setText("0.20")

        self.cov_cutoff_value_label = QLabel("cutoff: —")
        self.cov_quantile_value_label = QLabel("quantile: —%")

        _cov_row = QWidget()
        _cov_row_layout = QHBoxLayout(_cov_row)
        _cov_row_layout.setContentsMargins(0, 0, 0, 0)
        _cov_row_layout.addWidget(self.periodicity_threshold_cov_LineEdit)
        _cov_row_layout.addSpacing(12)
        _cov_row_layout.addWidget(self.cov_cutoff_value_label)
        _cov_row_layout.addSpacing(12)
        _cov_row_layout.addWidget(self.cov_quantile_value_label)
        _cov_row_layout.addStretch(1)
        periodicity_layout.addRow("CoV threshold:", _cov_row)

        # Row 3: action button + toggle + display checkbox
        _periodic_btn_row = QWidget()
        _periodic_btn_row_layout = QHBoxLayout(_periodic_btn_row)
        _periodic_btn_row_layout.setContentsMargins(0, 0, 0, 0)

        self.periodicity_button = QPushButton("Find Periodic Cells")
        self.periodicity_toggle = ToggleSwitch(label_on="CoV", label_off="STD")
        self.periodic_display_checkbox = QCheckBox("Show Cells")
        self.periodic_display_checkbox.setDisabled(True)

        _periodic_btn_row_layout.addWidget(self.periodicity_button)
        _periodic_btn_row_layout.addWidget(self.periodicity_toggle)
        _periodic_btn_row_layout.addWidget(self.periodic_display_checkbox)
        _periodic_btn_row_layout.addStretch(1)

        periodicity_layout.addRow(_periodic_btn_row)

        # Row 4: labels for % periodic cells (STD / CoV)
        self.std_periodic_per_active_label = QLabel("Cells below STD cutoff: — %")
        self.cov_periodic_per_active_label = QLabel("Cells below CoV cutoff: — %")

        _pct_row = QWidget()
        _pct_row_layout = QHBoxLayout(_pct_row)
        _pct_row_layout.setContentsMargins(0, 0, 0, 0)
        _pct_row_layout.addWidget(self.std_periodic_per_active_label)
        _pct_row_layout.addSpacing(24)
        _pct_row_layout.addWidget(self.cov_periodic_per_active_label)
        _pct_row_layout.addStretch(1)

        periodicity_layout.addRow(_pct_row)

        # Add the form container to the collapsible box
        periodicity_group.add_widget(form_container)
        return periodicity_group

    def _build_synchronicity_group(self) -> CollapsibleBox:
        self.synch_group = CollapsibleBox("Synchronicity Analysis")

        # Container widget for the QFormLayout
        form_container = QWidget()
        synch_layout = QFormLayout(form_container)
        synch_layout.setContentsMargins(0, 0, 0, 0)
        synch_layout.setSpacing(5)

        # Neighbour Distance input
        self.neighbour_distance_spin = _double_spin_box(
            decimals=2, minimum=0, maximum=100, value=10.0
        )
        synch_layout.addRow("Neighbour Distance [µm]:", self.neighbour_distance_spin)

        # Time Window input
        self.time_window_spin = _double_spin_box(
            decimals=2, minimum=0, maximum=100, value=10.0
        )
        synch_layout.addRow("Time Window [s]:", self.time_window_spin)

        # Action buttons
        _btn_row = QWidget()
        _btn_row_layout = QHBoxLayout(_btn_row)
        _btn_row_layout.setContentsMargins(0, 0, 0, 0)

        self.show_neighbour_btn = QPushButton("Show Neighbours")
        self.calculate_synchronous_btn = QPushButton("Calculate Synchronicity")

        _btn_row_layout.addWidget(self.show_neighbour_btn)
        _btn_row_layout.addWidget(self.calculate_synchronous_btn)
        _btn_row_layout.addStretch(1)

        synch_layout.addRow(_btn_row)

        # Optional labels for results
        self.synch_results_label = QLabel("")
        synch_layout.addRow("Results:", self.synch_results_label)

        # Add the form container to the collapsible box
        self.synch_group.add_widget(form_container)
        return self.synch_group

    def _build_save_group(self) -> QGroupBox:
        save_group = QGroupBox("Saving")
        save_layout = QVBoxLayout()
        save_row_layout = QHBoxLayout()

        self.overwrite_checkbox = QCheckBox("Overwrite")
        self.overwrite_checkbox.setChecked(True)
        self.save_btn = QPushButton("Save Analysis")
        self.save_btn.clicked.connect(self._save_analysis_results)

        save_row_layout.addWidget(self.save_btn)
        save_row_layout.addWidget(self.overwrite_checkbox)

        save_layout.addLayout(save_row_layout)
        save_group.setLayout(save_layout)
        return save_group

    def _create_menu_bar(self):
        self.menu_bar = self.menuBar()

        file_menu = self.menu_bar.addMenu("File")

        restart_action = QAction("Open New Image", self)
        restart_action.triggered.connect(self._on_new_analysis)
        file_menu.addAction(restart_action)

        concat_action = QAction("Generate Experiment Overview CSV", self)
        concat_action.triggered.connect(self._on_concat_results)
        file_menu.addAction(concat_action)

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

    def on_slider_moved(self, idx):
        self.current_frame = idx
        self.image_display.display_dask_frame(
            self.frames[self.current_frame, self.display_channel, :, :]
        )
        # only the frame marker moves, so the trace is not plotted again
        self.frame_marker.set_xdata([idx, idx])
        self.canvas.draw_idle()

    def roi_selected(self, idx):
        self.selected_roi = idx
        self.update_intensity_trace()

    def update_intensity_trace(self):
        self.ax.clear()
        color_raw = "#1E2B33"
        color_detrended = "#4C0308"

        if self.show_detrended:
            try:
                self.ax.plot(
                    self.detrended_intensities[:, self.selected_roi],
                    color=color_detrended,
                    label=f"ROI {self.selected_roi}",
                )
                self.ax.set_ylabel("Detrended Intensity")
            except Exception as e:
                logger.error(f"Error plotting detrended intensities: {e}")
        else:
            self.ax.plot(
                self.intensities[:, self.selected_roi],
                color=color_raw,
                label=f"ROI {self.selected_roi}",
            )
            self.ax.set_ylabel("Raw Intensity")

        if self.peaks is not None:
            peak_indices = np.where(self.peaks[:, self.selected_roi] == 1)[0]
            self.ax.plot(
                peak_indices,
                (
                    self.detrended_intensities
                    if self.show_detrended
                    else self.intensities
                )[peak_indices, self.selected_roi],
                "bx",
                markersize=4,
            )

        self.frame_marker = self.ax.axvline(
            self.current_frame, color="k", linestyle="--"
        )
        self.ax.set_xlabel("Frame")
        self.ax.legend()
        self.canvas.draw()

    def show_raw_traces(self):
        self.show_detrended = False
        self.update_intensity_trace()

    def _typed_frame_interval(self) -> float:
        """
        Returns the frame interval of the spin box. The box can only show it rounded, so
        as long as the user has not changed it, the exact frame interval is returned.
        """
        typed = self.frame_interval_spin.value()
        exact = self.analysis_output.frame_interval
        if typed == round(exact, FRAME_INTERVAL_DECIMALS):
            return exact
        return typed

    def show_detrended_traces(self):
        sinc_value = self.sinc_filter_spin.value()
        frame_interval = self._typed_frame_interval()
        # If needed, recompute
        needs_recompute = (
            self.detrended_intensities is None
            or self.last_sinc_window != sinc_value
            or self.analysis_output.frame_interval != frame_interval
        )
        if needs_recompute:
            if self._detrending:
                return  # a detrend is already running and will update the trace
            self._start_detrend_worker(sinc_value, frame_interval)
        else:
            self.show_detrended = True
            self.update_intensity_trace()

    def _periodic_cell_highlight(self):
        if self.periodic_display_checkbox.isChecked():
            if self.analysis_output is not None:
                if (
                    self.periodicity_toggle.value() == "STD"
                    and self.experimental_stds is not None
                ):
                    periodic_indices = np.where(
                        self.experimental_stds < self.analysis_output.std_cutoff
                    )[0].tolist()
                elif (
                    self.periodicity_toggle.value() == "CoV"
                    and self.experimental_covs is not None
                ):
                    periodic_indices = np.where(
                        self.experimental_covs < self.analysis_output.cov_cutoff
                    )[0].tolist()
                else:
                    periodic_indices = []
                self.image_display.draw_roi_centers(periodic_indices, radius=5)
        else:
            self.image_display.clear_roi_centers()

    def _start_detrend_worker(self, sinc_value, frame_interval):
        self._detrending = True
        # the window the traces are detrended with, it may be changed while this runs
        self._detrend_sinc_window = sinc_value
        if frame_interval != self.analysis_output.frame_interval:
            self.analysis_output.frame_interval_source = "parameter"
        self.analysis_output.frame_interval = frame_interval
        self.detrend_worker = FunctionWorker(
            detrend_with_sinc_filter,
            signals=self.intensities,
            cutoff_period=sinc_value,
            sampling_interval=self.analysis_output.frame_interval,
            description="Detrending",
        )
        start_worker(self, self.detrend_worker, self._detrend_done, self._detrend_error)

    def _detrend_done(self, detrended):
        self._detrending = False
        self.detrended_intensities = detrended
        self.analysis_params.sinc_filter_window = self._detrend_sinc_window
        self.last_sinc_window = self._detrend_sinc_window
        # Optionally hide progress bar or message here
        self.show_detrended = True
        self.update_intensity_trace()

    def _detrend_error(self, error):
        self._detrending = False
        QMessageBox.critical(self, "Error", f"Detrending failed: {error}")

    def _start_peak_worker(self):
        if self.detrended_intensities is None:
            QMessageBox.warning(
                self,
                "Missing Data",
                "Detrended traces are required. Please detrend first.",
            )
            return

        # update analysis params
        self.analysis_params.peak_min_width = self.peak_min_width_spin.value()
        self.analysis_params.peak_max_width = self.peak_max_width_spin.value()
        self.analysis_params.peak_prominence = self.peak_prominence_spin.value()
        self.analysis_params.peak_min_height = self.peak_min_height_spin.value()

        # Convert time-based spinboxes to frame units, the same way as the pipeline
        min_width, max_width = peak_widths_in_frames(
            self.analysis_params, self.analysis_output
        )

        self.find_peaks_btn.setEnabled(False)  # Prevent repeat clicks

        # Launch async worker
        self.peak_worker = FunctionWorker(
            do_peak_analysis,
            self.detrended_intensities,
            min_width,
            max_width,
            self.analysis_params,
            self.analysis_output,
            description="Peak Detection",
        )
        start_worker(self, self.peak_worker, self._peaks_detected, self._peak_error)

    def _peaks_detected(self, result):
        self.peaks, _, _, self.analysis_output = result
        self.find_peaks_btn.setEnabled(True)

        percentage_four_peaks = (
            self.analysis_output.cells_four_peaks
            / self.analysis_output.number_of_cells
            * 100
        )

        # display the peak stats
        self.norm_peak_label.setText(f"{self.analysis_output.normalized_peaks:.2f}")
        self.fourpeak_label.setText(f"{percentage_four_peaks:.0f}%")
        self.frequency_label.setText(f"{self.analysis_output.median_frequency_mHz:.2f}")

        # update the intensity trace to show peaks
        self.update_intensity_trace()

    def _peak_error(self, error):
        QMessageBox.critical(self, "Error", f"Peak detection failed: {error}")
        self.find_peaks_btn.setEnabled(True)

    def _start_periodicity_worker(self):
        if self.peaks is None:
            QMessageBox.warning(
                self, "Missing Data", "Run peak detection before computing periodicity."
            )
            return

        for line_edit, name in (
            (self.periodicity_threshold_std_LineEdit, "STD threshold (0 to 1000)"),
            (self.periodicity_threshold_cov_LineEdit, "CoV threshold (0 to 1)"),
        ):
            if not line_edit.hasAcceptableInput():
                QMessageBox.warning(
                    self, "Invalid Input", f"Please enter a number for the {name}."
                )
                return

        self.analysis_params.periodicity_method = (
            "cutoff" if self.rb_periodicity_cutoff.isChecked() else "quantile"
        )
        self.analysis_params.std_threshold = float(
            self.periodicity_threshold_std_LineEdit.text()
        )
        self.analysis_params.cov_threshold = float(
            self.periodicity_threshold_cov_LineEdit.text()
        )

        self.periodicity_button.setEnabled(False)
        self.periodicity_worker = FunctionWorker(
            _find_periodic_cells,
            self.peaks,
            self.analysis_params,
            self.analysis_output,
            description="Periodicity Analysis",
        )
        start_worker(
            self,
            self.periodicity_worker,
            self._periodicity_done,
            self._periodicity_error,
        )

    def _periodicity_done(self, results):
        (
            self.analysis_output,
            self.experimental_stds,
            self.experimental_covs,
            self.roi_dataframe,
        ) = results

        # Update the labels with periodicity cutoffs and quantiles
        self.cov_cutoff_value_label.setText(
            f"cutoff: {self.analysis_output.cov_cutoff:.2f}"
        )
        self.cov_quantile_value_label.setText(
            f"quantile: {(self.analysis_output.cov_quantile) * 100:.2f}%"
        )
        self.std_cutoff_value_label.setText(
            f"cutoff: {self.analysis_output.std_cutoff:.2f}"
        )
        self.std_quantile_value_label.setText(
            f"quantile: {(self.analysis_output.std_quantile) * 100:.2f}%"
        )

        # Update the periodicity percentage labels
        n_cells = self.analysis_output.number_of_cells
        std_periodic_per_all = round(
            self.analysis_output.experimental_below_std / n_cells * 100, 1
        )
        cov_periodic_per_all = round(
            self.analysis_output.experimental_below_cov / n_cells * 100, 1
        )
        self.std_periodic_per_active_label.setText(
            f"Cells below STD cutoff: {std_periodic_per_all}%"
        )
        self.cov_periodic_per_active_label.setText(
            f"Cells below CoV cutoff: {cov_periodic_per_all}%"
        )

        self.periodicity_button.setEnabled(True)
        self.periodic_display_checkbox.setEnabled(True)

    def _periodicity_error(self, error):
        QMessageBox.critical(self, "Error", f"Periodic ROI analysis failed: {error}")
        self.periodicity_button.setEnabled(True)

    def _start_neighbours_worker(self):
        if self.peaks is None:
            QMessageBox.warning(
                self, "Missing Data", "Run peak detection before showing neighbours."
            )
            return
        if self.selected_roi is None:
            QMessageBox.warning(self, "No ROI Selected", "Please select an ROI first.")
            return

        self.show_neighbour_btn.setEnabled(False)
        self.analysis_params.neighbour_tolerance = self.neighbour_distance_spin.value()
        neighbour_tolerance_pixels = neighbour_tolerance_in_pixels(
            self.analysis_params, self.analysis_output
        )
        self.neighbours_worker = FunctionWorker(
            get_adjacency_matrix,
            self.masks,
            neighbour_tolerance=neighbour_tolerance_pixels,
            description="Neighbourhood Analysis",
        )
        start_worker(
            self, self.neighbours_worker, self._neighbours_done, self._neighbours_error
        )

    def _neighbours_done(self, results):
        self.adjacency_matrix, self.shortest_path_matrix = results

        # Highlight neighbours of the selected ROI
        self.image_display.draw_roi_connections(self.shortest_path_matrix)

        self.show_neighbour_btn.setEnabled(True)

    def _neighbours_error(self, error):
        QMessageBox.critical(self, "Error", f"Finding neighbours failed: {error}")
        self.show_neighbour_btn.setEnabled(True)

    def _start_synchronicity_worker(self):
        if self.peaks is None:
            QMessageBox.warning(
                self,
                "Missing Data",
                "Run peak detection before calculating synchronicity.",
            )
            return
        if self.adjacency_matrix is None:
            QMessageBox.warning(
                self, "No Neighbour Data", "Please find neighbours first."
            )
            return

        self.calculate_synchronous_btn.setEnabled(False)
        self.analysis_params.time_window = self.time_window_spin.value()
        self.synchronicity_worker = FunctionWorker(
            calculate_synchronicity,
            self.shortest_path_matrix,
            self.peaks,
            self.analysis_params,
            self.analysis_output,
            description="Synchronicity Analysis",
        )
        start_worker(
            self,
            self.synchronicity_worker,
            self._synchronicity_done,
            self._synchronicity_error,
        )

    def _synchronicity_done(self, results):
        self.analysis_output = results[0]
        self.true_synchronous_peaks_matrix = results[1]
        self.synch_results_label.setText(
            f"# of True Synchronous Peaks: {results[1].sum()}\n"
            f"# of Random Synchronous Peaks: {results[2].sum()}\n"
            f"Z-Score: {self.analysis_output.experiment_1st_neighbour_zscore:.2f}"
        )

        self.calculate_synchronous_btn.setEnabled(True)

    def _synchronicity_error(self, error):
        tb_str = "".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        )
        QMessageBox.critical(self, "Error", f"Synchronicity analysis failed: {tb_str}")
        self.calculate_synchronous_btn.setEnabled(True)

    def _save_analysis_results(self):
        if self.true_synchronous_peaks_matrix is None or self.experimental_stds is None:
            QMessageBox.warning(
                self,
                "Missing Data",
                "Find the periodic cells and calculate the synchronicity before saving.",
            )
            return
        self.save_btn.setEnabled(False)
        self.analysis_worker = FunctionWorker(
            _save_with_projection,
            tproj_path=self.analysis_output.tproj_path,
            masks=self.masks,
            synchronous_peaks_matrix=self.true_synchronous_peaks_matrix,
            analysis_output=self.analysis_output,
            analysis_params=self.analysis_params,
            peaks=self.peaks,
            detrended_intensities=self.detrended_intensities,
            intensities=self.intensities,
            stds=self.experimental_stds,
            covs=self.experimental_covs,
            roi_dataframe=self.roi_dataframe,
            overwrite=self.overwrite_checkbox.isChecked(),
            description="Saving",
        )
        start_worker(
            self, self.analysis_worker, self.on_save_finished, self.on_save_error
        )

    def on_save_finished(self, _):
        self.save_btn.setEnabled(True)
        QMessageBox.information(
            self, "Save Complete", "Analysis results saved successfully."
        )

    def on_save_error(self, e: Exception):
        self.save_btn.setEnabled(True)
        QMessageBox.critical(self, "Save Error", str(e))

    def closeEvent(self, event):
        if defer_close_while_running(self, event):
            return
        # a running QThread that is destroyed with its window aborts the application
        stop_workers(self)
        super().closeEvent(event)

    def _on_new_analysis(self):
        if workers_running(self):
            QMessageBox.information(
                self,
                "Analysis Running",
                "Please wait until the running analysis step has finished.",
            )
            return
        if self.app_controller:
            self.app_controller.restart()

    def _on_concat_results(self):
        try:
            if not self.analysis_output.filepath:
                QMessageBox.warning(self, "No Image", "No image is currently open.")
                return

            # Get parent folder of the current image
            image_path = Path(self.analysis_output.filepath)
            root_folder = image_path.parent

            # Call your concatenation function
            concat_analysis_files(str(root_folder))

            QMessageBox.information(
                self,
                "Success",
                f"Analysis results concatenated successfully in:\n{root_folder / 'experiment_overview.csv'}",
            )
        except FileNotFoundError as e:
            QMessageBox.warning(self, "No CSV Files", str(e))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred:\n{e}")
