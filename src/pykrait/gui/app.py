import logging
import sys
import os
import warnings
from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication
from pykrait.gui.fileselection import FileSelectionWindow
from pykrait.gui.mainwindow import MainWindow
from pykrait.io.files import get_pykrait_version

logger = logging.getLogger(__name__)


def warning_no_source_line(message, category, filename, lineno, file=None, line=None):
    return f"{filename}:{lineno}: {category.__name__}: {message}\n"


class AppController:
    def __init__(self):
        # format warnings to not print the source line every time
        warnings.formatwarning = warning_no_source_line

        self.app = QApplication(sys.argv)

        # Loading icon from resources
        appIcon = QIcon()
        try:
            from importlib.resources import files

            icon_path = files("pykrait.gui.resources").joinpath("logo.png")
        except Exception:
            icon_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "pykrait/gui/resources",
                    "logo.png",
                )
            )
        if os.path.exists(icon_path):
            for size in (16, 24, 32, 48, 64, 256):
                appIcon.addFile(str(icon_path), QSize(size, size))
            self.app.setWindowIcon(appIcon)
        else:
            logger.warning(f"Warning: Icon file not found at {icon_path}")

        # Set application name and version
        self.app.setApplicationName("pyKrait")
        version = get_pykrait_version()
        if version is not None:
            self.app.setApplicationVersion(version)
        self._open_file_selection()

        # updating the style
        self.app.setStyle("Fusion")
        light_palette = self.app.style().standardPalette()
        self.app.setPalette(light_palette)

        # Keep references to windows
        self.main_window = None

    def on_analysis_complete(self, results):
        # Called when analysis is done in FileSelectionWindow
        # Hide file selection window
        self.file_selection.close()
        # Launch main window, keep reference
        self.main_window = MainWindow(
            frames=results["frames"],
            display_channel=results.get("intensity_channel", 0),
            mask=results["masks"],
            rois=results["rois"],
            mean_intensities=results["mean_intensities"],
            analysis_output=results["analysis_output"],
            analysis_params=results["analysis_parameters"],
            app_controller=self,
        )
        self.main_window.show()

    def _open_file_selection(self):
        self.file_selection = FileSelectionWindow()
        self.file_selection.analysis_complete.connect(self.on_analysis_complete)
        self.file_selection.show()

    def run(self):
        sys.exit(self.app.exec())

    def restart(self):
        if self.main_window:
            self.main_window.close()
            self.main_window = None

        self._open_file_selection()
