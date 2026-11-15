import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

import dask.array as da  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from pykrait.gui.mainwindow import ImageDisplay  # noqa: E402
from pykrait.io.images import get_pixel_contour_from_label_img  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize(
    "shape", [(1024, 1024), (1000, 1000), (1200, 1600), (400, 400), (512, 1024)]
)
def test_roi_outline_matches_displayed_frame(app, shape):
    h, w = shape
    masks = np.zeros(shape, dtype=np.uint16)
    top, left, size = h // 2, w - 80, 40
    masks[top : top + size, left : left + size] = 1

    display = ImageDisplay()
    display.display_dask_frame(da.from_array(masks.astype(float)))
    rois = [
        [(x + 0.5, y + 0.5) for x, y in poly]
        for poly in get_pixel_contour_from_label_img(masks)
    ]
    display.set_rois(rois)

    # the displayed frame covers the original pixel grid, apart from trimmed edge pixels
    frame_rect = display.image_item.sceneBoundingRect()
    assert frame_rect.left() == 0 and frame_rect.top() == 0
    assert w - display.display_factor < frame_rect.width() <= w
    assert h - display.display_factor < frame_rect.height() <= h

    # the outline sits on the border of the masked square in scene coordinates
    roi_rect = display.roi_items[0].sceneBoundingRect()
    assert roi_rect.left() == pytest.approx(left, abs=1)
    assert roi_rect.top() == pytest.approx(top, abs=1)
    assert roi_rect.width() == pytest.approx(size, abs=1)
    assert roi_rect.height() == pytest.approx(size, abs=1)


@pytest.mark.parametrize(
    "matrix, direction",
    [
        ([[0, 1], [0, 0]], "0->1"),
        ([[0, 0], [1, 0]], "1->0"),
        ([[0, 2], [3, 0]], "both"),
    ],
)
def test_synchronicity_image_draws_pairs_counted_in_either_triangle(matrix, direction):
    from unittest.mock import patch

    import matplotlib.pyplot as plt

    from pykrait.visualization import image_overlays

    masks = np.zeros((40, 80), dtype=np.uint16)
    masks[10:30, 5:25] = 1
    masks[10:30, 55:75] = 2
    with patch.object(image_overlays, "draw_arrow") as draw_arrow:
        figure = image_overlays.create_synchronicity_image(
            np.zeros((40, 80)), masks, np.array(matrix)
        )
    plt.close(figure)
    assert draw_arrow.call_count == (2 if direction == "both" else 1)


def test_frame_interval_box_keeps_the_exact_interval_until_it_is_changed(app):
    from types import SimpleNamespace

    from pykrait.gui.mainwindow import MainWindow, _double_spin_box

    window = SimpleNamespace(
        analysis_output=SimpleNamespace(frame_interval=1 / 30),
        frame_interval_spin=_double_spin_box(4, 0.0001, 1000, 1 / 30),
    )
    # the box shows 0.0333, the analysis keeps 1/30
    assert MainWindow._typed_frame_interval(window) == 1 / 30
    window.frame_interval_spin.setValue(0.05)
    assert MainWindow._typed_frame_interval(window) == 0.05
