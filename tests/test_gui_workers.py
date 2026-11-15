import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QObject, Signal, Slot  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from pykrait.gui.async_workers import start_worker, stop_workers  # noqa: E402


class _Worker(QObject):
    finished = Signal(int)
    error = Signal(str)

    def __init__(self, fail: bool):
        super().__init__()
        self.fail = fail

    @Slot()
    def run(self):
        if self.fail:
            self.error.emit("boom")
        else:
            self.finished.emit(42)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def _wait_until_stopped(app, owner, timeout_s=5):
    import time

    deadline = time.monotonic() + timeout_s
    while owner._worker_threads and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


@pytest.mark.parametrize("fail", [False, True])
def test_worker_thread_stops_on_finish_and_error(app, fail):
    owner = QWidget()
    results, errors = [], []
    start_worker(owner, _Worker(fail), results.append, errors.append)
    _wait_until_stopped(app, owner)

    # the thread must stop in both cases, otherwise closing the window aborts the app
    assert owner._worker_threads == []
    assert (errors, results) == ((["boom"], []) if fail else ([], [42]))


def test_stop_workers_waits_for_running_threads(app):
    owner = QWidget()
    for _ in range(3):
        start_worker(owner, _Worker(True), lambda *_: None, lambda *_: None)
    stop_workers(owner)
    owner.deleteLater()
    app.processEvents()


class _FakeThread:
    def __init__(self):
        self.running = True

    def isRunning(self):
        return self.running


class _SpontaneousClose:
    def __init__(self):
        self.ignored = False

    def spontaneous(self):
        return True

    def ignore(self):
        self.ignored = True


@pytest.mark.parametrize("answer", ["Yes", "No"])
def test_closing_during_an_analysis_does_not_freeze(app, monkeypatch, answer):
    import time

    from PySide6.QtWidgets import QMessageBox

    import pykrait.gui.async_workers as async_workers

    button = getattr(QMessageBox.StandardButton, answer)
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: button)
    quit_calls = []
    monkeypatch.setattr(app, "quit", lambda: quit_calls.append(True))

    window = QWidget()
    window.show()
    thread = _FakeThread()
    window._worker_threads = [(thread, None)]
    event = _SpontaneousClose()

    # returns at once instead of waiting for the thread
    assert async_workers.defer_close_while_running(window, event)
    assert event.ignored
    if answer == "No":
        assert window.isVisible()
        window._worker_threads = []
        window.close()
        return

    assert not window.isVisible() and window.close_when_finished
    thread.running = False
    deadline = time.monotonic() + 5
    while not quit_calls and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    # closes itself and quits once the analysis has finished, if no window is left
    assert quit_calls


def test_closing_without_running_analysis_or_by_the_program_is_not_deferred(app):
    from pykrait.gui.async_workers import defer_close_while_running

    window = QWidget()
    window._worker_threads = [(_FakeThread(), None)]

    class ProgramClose(_SpontaneousClose):
        def spontaneous(self):
            return False

    assert not defer_close_while_running(window, ProgramClose())
    window._worker_threads = []
    assert not defer_close_while_running(window, _SpontaneousClose())


def test_csv_mode_saves_a_projection_for_the_results(app, tmp_path):
    import numpy as np
    import tifffile

    from pykrait.gui.async_workers import ExtractIntensitiesWorker
    from pykrait.io.files import save_Txnrois
    from pykrait.pipeline.pipeline import AnalysisOutput, AnalysisParameters

    video = np.random.default_rng(0).integers(0, 1000, (10, 16, 16), dtype=np.uint16)
    video_path = tmp_path / "video.tif"
    tifffile.imwrite(video_path, video, imagej=True, metadata={"axes": "TYX"})
    masks = np.zeros((16, 16), dtype=np.uint16)
    masks[2:8, 2:8] = 1
    masks_path = tmp_path / "masks.tif"
    tifffile.imwrite(masks_path, masks)
    intensities_path = tmp_path / "intensities.csv.zst"
    save_Txnrois(np.ones((10, 1)), 1.0, str(intensities_path))

    output = AnalysisOutput(filepath=str(video_path), filename="video.tif")
    output.masks_path = str(masks_path)
    worker = ExtractIntensitiesWorker(
        AnalysisParameters(frame_interval=1.0), output, "csv", str(intensities_path)
    )
    results, errors = [], []
    worker.finished.connect(results.append)
    worker.error.connect(errors.append)
    worker.run()

    assert errors == []
    tproj_path = results[0]["analysis_output"].tproj_path
    assert tproj_path is not None and os.path.exists(tproj_path)
