import os

import psutil
import pytest

import pykrait.io.images as images
import pykrait.pipeline.pipeline as pipeline


def _open_by_this_process(path: str) -> bool:
    target = os.path.normcase(os.path.realpath(path))
    return any(
        os.path.normcase(os.path.realpath(f.path)) == target
        for f in psutil.Process().open_files()
    )


class _WindowsFileLocking:
    """Stands in for `os` in the pipeline and refuses, like Windows, to replace or delete open files."""

    def __getattr__(self, name):
        return getattr(os, name)

    @staticmethod
    def _check(*paths):
        for path in paths:
            if os.path.exists(path) and _open_by_this_process(path):
                raise PermissionError(
                    f"[WinError 32] The process cannot access the file because it is being used by another process: {path!r}"
                )

    def replace(self, source, destination):
        self._check(source, destination)
        os.replace(source, destination)

    def remove(self, path):
        self._check(path)
        os.remove(path)


@pytest.fixture(autouse=True)
def windows_file_locking(monkeypatch):
    """
    Lets the pipeline only replace and delete files that are not open, as on Windows, so
    that handles kept open by lazily read videos fail on every platform and not only in the
    Windows CI jobs. The handles are released as they would be on Windows.
    """
    monkeypatch.setattr(pipeline, "os", _WindowsFileLocking())
    monkeypatch.setattr(images, "_RELEASE_TIFF_HANDLES", True)
