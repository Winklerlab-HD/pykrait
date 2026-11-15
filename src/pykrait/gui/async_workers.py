import os
import logging
import traceback
import tifffile
import numpy as np

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import QApplication, QMessageBox

from pykrait.io.images import read_label_image, get_pixel_contour_from_label_img
from pykrait.io.files import read_Txnrois, get_analysis_file
from pykrait.preprocessing.segmentation import timelapse_projection
from pykrait.preprocessing.timeseries_extraction import extract_mean_intensities

from pykrait.pipeline.pipeline import (
    AnalysisParameters,
    AnalysisOutput,
    create_analysis_folder,
    load_timelapse,
    apply_motion_correction,
    create_masks,
    save_params,
    intensity_channel_index,
    create_intensity_cache,
    require_rois,
)

logger = logging.getLogger(__name__)


def start_worker(owner: QObject, worker: QObject, on_finished, on_error) -> QThread:
    """
    runs the `run` slot of a worker on a new QThread. The thread is stopped whether the worker
    finishes or fails, and the worker and thread are deleted afterwards. Running threads are
    tracked on the owner so that `stop_workers` can wait for them before the owner closes,
    as Qt aborts the application when a running QThread is destroyed.

    :param owner: widget that starts the worker and receives its results
    :type owner: QObject
    :param worker: worker with a `run` slot and `finished` and `error` signals
    :type worker: QObject
    :param on_finished: slot connected to the worker's `finished` signal
    :param on_error: slot connected to the worker's `error` signal
    :return: returns the started thread
    :rtype: QThread
    """
    if not hasattr(owner, "_worker_threads"):
        owner._worker_threads = []
    active = owner._worker_threads

    thread = QThread(owner)
    worker.moveToThread(thread)
    # holds a reference to the worker as well, so it is not garbage collected while running
    entry = (thread, worker)
    active.append(entry)

    thread.started.connect(worker.run)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    thread.finished.connect(lambda: entry in active and active.remove(entry))
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread


def stop_workers(owner: QObject) -> None:
    """
    waits for all worker threads started by `owner` with `start_worker` to finish.

    :param owner: widget that started the workers
    :type owner: QObject
    """
    for thread, _ in list(getattr(owner, "_worker_threads", [])):
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait()
        except RuntimeError:
            pass  # Qt already deleted the thread


def workers_running(owner: QObject) -> bool:
    """Returns whether any worker thread started by `owner` is still running."""
    for thread, _ in list(getattr(owner, "_worker_threads", [])):
        try:
            if thread.isRunning():
                return True
        except RuntimeError:
            pass  # Qt already deleted the thread
    return False


def defer_close_while_running(window, event) -> bool:
    """
    Keeps a window from freezing when the user closes it during an analysis. The running
    steps cannot be interrupted, so the user is asked whether to close the window as soon
    as they have finished; the window is then hidden and closes itself, quitting the
    application if no other window is open. Call it first in `closeEvent`.

    Windows closed by the program, e.g. the file selection once its analysis is done,
    close immediately.

    :param window: window that started its workers with `start_worker`
    :param event: the close event
    :return: returns True if the close was deferred or cancelled and `closeEvent` should return
    :rtype: bool
    """
    if not event.spontaneous() or not workers_running(window):
        return False
    event.ignore()
    answer = QMessageBox.question(
        window,
        "Analysis Running",
        "An analysis step is still running and cannot be interrupted.\n\n"
        "Close the window as soon as it has finished?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.Yes,
    )
    if answer != QMessageBox.StandardButton.Yes:
        return True

    window.close_when_finished = True
    window.hide()
    timer = QTimer(window)
    timer.setInterval(200)

    def close_if_finished():
        if workers_running(window):
            return
        timer.stop()
        window.close()
        # a hidden last window does not quit the application when it closes
        app = QApplication.instance()
        if app is not None and not any(
            widget.isVisible() for widget in app.topLevelWidgets()
        ):
            app.quit()

    timer.timeout.connect(close_if_finished)
    timer.start()
    return True


class ExtractIntensitiesWorker(QObject):
    """
    Worker for extracting mean intensities from a timelapse dataset, using either Cellpose segmentation or a label image.
    """

    progress_changed = Signal(int, str)  # progress percent, status message
    finished = Signal(dict)  # analysis results
    error = Signal(str)  # error message

    def __init__(
        self,
        analysis_params: AnalysisParameters,
        output_params: AnalysisOutput,
        mode: str,
        mean_intensities_path: str = None,
        save_temp_file: bool = True,
    ):
        super().__init__()
        self.analysis_parameters = analysis_params
        self.analysis_output = output_params
        self.mode = mode
        self.mean_intensities_path = mean_intensities_path
        # whether the intensity channel may be kept in a temporary file on disk
        self.save_temp_file = save_temp_file

    @Slot()
    def run(self):
        frame_cache = None
        try:
            self.analysis_output = create_analysis_folder(self.analysis_output)
            logger.info(f"Running analysis on {self.analysis_output.filename}")
            # loading the timelapse data and analysing frame interval
            self.timelapse_data, self.analysis_output = load_timelapse(
                self.analysis_output, self.analysis_parameters
            )
            tproj_file = get_analysis_file(
                self.analysis_output.filename,
                "tproj.ome.tif",
                self.analysis_output.analysis_folder,
            )
            if (
                self.mode in ("label_image", "csv")
                and self.analysis_output.tproj_path is None
                and os.path.exists(tproj_file)
            ):
                # the projection is only shown and saved with the results, so one from a
                # previous analysis spares reading the video for it
                self.analysis_output.tproj_path = tproj_file

            # the first pass over the video, the motion correction if it runs and the
            # projection otherwise, keeps the intensity channel, so the intensities do
            # not read the video again
            if (
                self.analysis_parameters.motion_correction
                or self.mode == "cellpose"
                or (
                    self.mode == "label_image"
                    and self.analysis_output.tproj_path is None
                )
            ):
                frame_cache = create_intensity_cache(
                    self.timelapse_data,
                    self.analysis_output,
                    self.analysis_parameters,
                    save_temp_file=self.save_temp_file,
                )

            if self.analysis_parameters.motion_correction:
                self.progress_changed.emit(5, "Correcting lateral motion...")
            (
                self.timelapse_data,
                self.analysis_output,
                raw_projection,
            ) = apply_motion_correction(
                self.timelapse_data,
                self.analysis_output,
                self.analysis_parameters,
                frame_cache=frame_cache,
            )

            # the intensity channel is given for the original video, which may differ
            # from the motion-corrected one if its reference channel was dropped
            channel_index = intensity_channel_index(
                self.analysis_parameters,
                self.analysis_output.number_of_channels,
                self.timelapse_data.shape[1],
            )

            projection_reads_video = (
                self.mode == "cellpose" and raw_projection is None
            ) or (
                self.mode == "label_image" and self.analysis_output.tproj_path is None
            )
            if (
                frame_cache is not None
                and raw_projection is None
                and not projection_reads_video
            ):
                # neither the correction nor the projection reads the video, so nothing
                # fills the cache and the intensities read the video directly
                frame_cache.close()
                frame_cache = None
            # the intensities are extracted from the kept channel where possible; the
            # cache is only read once the projection has filled it
            intensity_source = (
                self.timelapse_data
                if frame_cache is None
                else frame_cache.as_timelapse()
            )

            if self.mode == "cellpose":
                # perform the tproj
                self.progress_changed.emit(
                    10, "Performing T Projection and Cellpose Segmentation..."
                )
                # creating the masks and saving them to the analysis folder
                _, contours, self.masks, self.analysis_output = create_masks(
                    self.timelapse_data,
                    self.analysis_output,
                    self.analysis_parameters,
                    raw_projection=raw_projection,
                    frame_cache=frame_cache,
                )
                require_rois(self.masks, self.analysis_output.filename)
                # extracts the mean intensities of the masks
                self.progress_changed.emit(60, "Extracting intensities...")
                mean_intensities = extract_mean_intensities(
                    intensity_source,
                    self.masks,
                    channel_index=channel_index,
                )
            elif self.mode == "label_image":
                contours = None
                if self.analysis_output.tproj_path is None:
                    self._save_projection(tproj_file, frame_cache)
                self.progress_changed.emit(20, "Loading label image...")
                self.masks = read_label_image(self.analysis_output.masks_path)
                require_rois(self.masks, self.analysis_output.filename)
                # extracts the mean intensities of the masks
                self.progress_changed.emit(60, "Extracting intensities...")
                mean_intensities = extract_mean_intensities(
                    intensity_source,
                    self.masks,
                    channel_index=channel_index,
                )
            elif self.mode == "csv":
                contours = None
                self.progress_changed.emit(20, "Loading label image...")
                self.masks = read_label_image(self.analysis_output.masks_path)
                require_rois(self.masks, self.analysis_output.filename)
                self.progress_changed.emit(40, "Reading intensities...")
                mean_intensities = read_Txnrois(
                    self.mean_intensities_path,
                    n_frames=self.timelapse_data.shape[0],
                    n_rois=self.masks.max(),
                )
                if self.analysis_output.tproj_path is None:
                    # the results are saved on top of the projection
                    self.progress_changed.emit(60, "Performing T Projection...")
                    self._save_projection(tproj_file, None)
            else:
                raise ValueError("Unknown mode.")

            (
                self.analysis_output.number_of_frames,
                self.analysis_output.number_of_cells,
            ) = mean_intensities.shape

            # All done
            self.progress_changed.emit(90, "Calculating ROI boundaries...")
            if (
                contours is None
            ):  # create_masks already returns the contours of new masks
                contours = get_pixel_contour_from_label_img(self.masks)
            # contours are in pixel index coordinates, the display scene puts pixel
            # centres at +0.5, so the outlines are shifted to sit on the cell borders
            rois = [[(x + 0.5, y + 0.5) for x, y in poly] for poly in contours]

            self.progress_changed.emit(100, "Done.")

            results = {
                "frames": self.timelapse_data,
                "intensity_channel": channel_index,
                "analysis_output": self.analysis_output,
                "analysis_parameters": self.analysis_parameters,
                "masks": self.masks,
                "rois": rois,
                "mean_intensities": mean_intensities,
            }
            save_params(self.analysis_output, self.analysis_parameters, overwrite=True)
            self.finished.emit(results)

        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error during analysis: {e}\n{tb}")
            self.error.emit(str(e))
        finally:
            if frame_cache is not None:
                frame_cache.close()

    def _save_projection(self, path: str, frame_cache) -> None:
        """Computes the STD projection of the video and saves it as the tproj of the analysis."""
        logger.info("Performing T Projection...")
        tproj = timelapse_projection(
            self.timelapse_data,
            method="std",
            normalize=True,
            verbose=True,
            frame_cache=frame_cache,
        )
        tifffile.imwrite(
            path,
            tproj.astype(np.uint16),
            metadata={"axes": "CYX"},
            compression="zlib",
        )
        self.analysis_output.tproj_path = path


class FunctionWorker(QObject):
    """
    Worker that runs a function with the given arguments and emits its return value.
    """

    finished = Signal(object)  # return value of the function
    error = Signal(Exception)

    def __init__(self, function, *args, description: str = "Analysis", **kwargs):
        """
        :param function: function to run on the worker thread
        :param args: positional arguments of the function
        :param description: name of the step for the error message printed to the console
        :type description: str, optional
        :param kwargs: keyword arguments of the function
        """
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.description = description

    @Slot()
    def run(self):
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error(f"Error during {self.description}: {e}\n{tb}")
            self.error.emit(e)
            return
        self.finished.emit(result)
