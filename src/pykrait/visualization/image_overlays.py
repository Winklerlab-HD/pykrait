import logging
import numpy as np
from matplotlib.axes import Axes
import matplotlib.patheffects as pe
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
from typing import Tuple

from pykrait.io.images import get_pixel_contour_from_label_img

logger = logging.getLogger(__name__)

Point = Tuple[int, int]
Segment = Tuple[Point, Point]


def compute_offset_lines(
    p1: Point, p2: Point, pixel_shift: int = 10
) -> Tuple[Segment, Segment]:
    """
    Computes two parallel lines offset by pixel_shift based on the points p1 and p2.

    Returns the endpoints of the two offset lines: (p1_for, p2_for, p1_rev, p2_rev)
    where p1_for and p2_for are the endpoints of the line offset in the "forward" direction (perpendicular to the line from p1 to p2),
    and p1_rev and p2_rev are the endpoints of the line offset in the "reverse" direction.


    :param p1: point 1 (x1, y1)
    :type p1: tuple
    :param p2: point 2 (x2, y2)
    :type p2: tuple
    :param pixel_shift: shift by which to separate the parallel lines, defaults to 10
    :type pixel_shift: int, optional
    :return: _description_
    :rtype: Tuple[tuple, tuple, tuple, tuple]
    """
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    # Vector along the line
    v = p2 - p1

    # Normalize perpendicular vector
    perp = np.array([-v[1], v[0]])
    perp = perp / np.linalg.norm(perp)

    # Compute shifted points
    p1_for, p2_for = p1 + (pixel_shift / 2) * perp, p2 + (pixel_shift / 2) * perp
    p1_rev, p2_rev = p1 - (pixel_shift / 2) * perp, p2 - (pixel_shift / 2) * perp

    return (p1_for, p2_for), (p1_rev, p2_rev)


def draw_arrow(
    ax: Axes,
    p1: tuple[float, float],
    p2: tuple[float, float],
    color="k",
    lw: int = 1,
    mutation_scale: int = 10,
):
    """
    Robust, backend-safe arrow drawing.
    Avoids FancyArrowPatch (can crash under Qt/Agg).
    """
    import numpy as np

    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    # Skip invalid or degenerate arrows
    if not np.isfinite(p1).all() or not np.isfinite(p2).all():
        logger.warning(f"Skipping arrow: invalid coordinates p1={p1}, p2={p2}")
        return
    if np.allclose(p1, p2):
        ax.plot(*p1, marker="o", color=color, markersize=lw * 2, zorder=2)
        return

    dx, dy = p2 - p1
    arrow = ax.arrow(
        p1[0],
        p1[1],
        dx,
        dy,
        lw=lw,
        head_width=mutation_scale * 0.8,
        head_length=mutation_scale * 1.5,
        length_includes_head=True,
        color=color,
        overhang=0.2,
        zorder=3,
    )

    arrow.set_path_effects([pe.withStroke(linewidth=lw + 0.2, foreground="white")])


def create_synchronicity_image(
    tproj: np.ndarray,
    masks: np.ndarray,
    synchronous_peaks_matrix: np.ndarray,
    max_value: int = 7,
    pixel_shift: int = 15,
    savepath: str = None,
) -> Figure:
    """
    _summary_

    :param tproj: image to superimpose synchronicity traces on, typically the temporal projection of the calcium video
    :type tproj: np.ndarray
    :param masks: mask image of the cell ROIs as label image
    :type masks: np.ndarray
    :param synchronous_peaks_matrix: matrix containing the number of synchronous peaks between each pair of cells
    :type synchronous_peaks_matrix: np.ndarray
    :param max_value: maximum value to display in the colormap, defaults to 6
    :type max_value: int, optional
    :param pixel_shift: shift by which to separate the parallel lines, defaults to 15
    :type pixel_shift: int, optional
    :param savepath: path where to save the generated image, defaults to None
    :type savepath: str, optional
    :return: the generated figure
    :rtype: Figure
    """
    if tproj.ndim >= 2:
        tproj = tproj.squeeze()
        if tproj.ndim != 2:
            raise ValueError(f"tproj must be a 2D array, not of shape {tproj.shape}.")

    fig = Figure()  # without pyplot, so it works with any backend and on worker threads
    ax = fig.subplots()
    ax.imshow(tproj, cmap="gray")

    # Plot cell boundaries
    rois = get_pixel_contour_from_label_img(masks)
    for poly in rois:
        xs, ys = zip(*poly)
        ax.plot(xs, ys, c="w", alpha=0.3, linestyle="-", linewidth=0.5, zorder=1)

    # colormap
    start_cmap = 0.5
    stop_cmap = 0.9

    intervals = max_value + 1
    bounds = np.arange(0, intervals + 1, 1)

    cm_subsection = np.linspace(start_cmap, stop_cmap, max_value)
    colors = [cm.magma(x) for x in cm_subsection]

    cmap = ListedColormap([(0.5, 0.5, 0.5, 1), *colors])
    norm = BoundaryNorm(bounds, ncolors=cmap.N, clip=False)

    # Find all pairs (i, j) with i < j that have synchronous peaks in either direction.
    # The counts of a pair can lie in the upper or the lower triangle, so both are checked
    centroids = np.array([np.mean(poly, axis=0) for poly in rois])
    synchronous_peaks_matrix = np.asarray(synchronous_peaks_matrix)
    either_direction = (synchronous_peaks_matrix > 0) | (synchronous_peaks_matrix.T > 0)
    idx_i, idx_j = np.where(np.triu(either_direction, 1))

    for i, j in zip(idx_i, idx_j):
        x1, y1 = centroids[i]
        x2, y2 = centroids[j]
        val_for, val_rev = (
            min(synchronous_peaks_matrix[i, j], max_value),
            min(synchronous_peaks_matrix[j, i], max_value),
        )
        if val_for > 0 and val_rev > 0:
            (p1_for, p2_for), (p1_rev, p2_rev) = compute_offset_lines(
                (x1, y1), (x2, y2), pixel_shift=pixel_shift
            )
            draw_arrow(
                ax, p1_for, p2_for, color=cmap(norm(val_for)), lw=2, mutation_scale=5
            )
            draw_arrow(
                ax, p2_rev, p1_rev, color=cmap(norm(val_rev)), lw=2, mutation_scale=5
            )
        elif val_for > 0:
            draw_arrow(
                ax,
                (x1, y1),
                (x2, y2),
                color=cmap(norm(val_for)),
                lw=2,
                mutation_scale=5,
            )
        elif val_rev > 0:
            draw_arrow(
                ax,
                (x2, y2),
                (x1, y1),
                color=cmap(norm(val_rev)),
                lw=2,
                mutation_scale=5,
            )
    ax.set_axis_off()
    ax.set_xlim(0, tproj.shape[1])  # width in x
    ax.set_ylim(tproj.shape[0], 0)  # height in y, inverted so (0,0) is top-left

    cbar = fig.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="vertical",
        spacing="proportional",
    )
    centers = 0.5 * (bounds[:-1] + bounds[1:])
    cbar.set_ticks(centers)
    labels = [str(int(b)) for b in bounds[:-1]]
    labels[-1] = f"{max_value}+"
    cbar.set_ticklabels(labels)

    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Number of Synchronous Events", fontsize=7)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding
    if savepath is not None:
        fig.savefig(savepath, dpi=300, transparent=True)
    return fig


def create_heatmap_image(
    masks: np.ndarray,
    peak_series: np.ndarray,
    cmap: str = "magma",
    maxpeaks: int = 20,
    savepath: str = None,
) -> Figure:
    """
    generates a heatmap for the underlying calcium video where each cell is colored according to its number of peaks.

    :param masks: mask image
    :type masks: np.ndarray
    :param peak_series: peak series of shape T x n_rois where 1 denotes a peak
    :type peak_series: np.ndarray
    :param cmap: colormap to draw, defaults to "magma"
    :type cmap: str, optional
    :param maxpeaks: upper bound for peak cutoff, defaults to 20
    :type maxpeaks: int, optional
    :param savepath: path where to save, defaults to None
    :type savepath: str, optional
    :return: returns the matplotlib figure
    :rtype: Figure
    """

    fig = Figure()  # without pyplot, so it works with any backend and on worker threads
    ax = fig.subplots()

    # colormap
    start = 0.2
    stop = 0.7
    intervals = maxpeaks + 1
    cm_subsection = np.linspace(start, stop, intervals)
    cmap_subsections = [cm.magma(x) for x in cm_subsection]
    colors = [(0, 0, 0, 1), (0.5, 0.5, 0.5, 1), *cmap_subsections[1:]]
    custom_cmap = ListedColormap(colors).with_extremes(over=cmap_subsections[-1])
    bounds = np.arange(0, maxpeaks + 3)
    norm = BoundaryNorm(bounds, custom_cmap.N)

    heatmap_image = np.zeros(masks.shape, dtype=np.uint8)
    for i in range(1, peak_series.shape[1] + 1):
        locs = np.where(masks == i)
        peak_count = np.sum(peak_series[:, i - 1])
        heatmap_image[locs] = min(
            peak_count + 1, maxpeaks + 1
        )  # +1 to account for background

    ax.imshow(heatmap_image, cmap=custom_cmap, norm=norm)

    # Plot cell boundaries
    rois = get_pixel_contour_from_label_img(masks)
    for poly in rois:
        xs, ys = zip(*poly)
        ax.plot(xs, ys, c="w", alpha=0.3, linestyle="-", linewidth=0.5, zorder=1)

    # adopting the
    ax.set_axis_off()
    ax.set_xlim(0, masks.shape[1])  # width in x
    ax.set_ylim(masks.shape[0], 0)  # height in y, inverted so (0,0) is top-left

    cbar_bounds = np.arange(1, maxpeaks + 3)
    cbar_cmap = ListedColormap(colors[1:])  # exclude black
    cbar_norm = BoundaryNorm(cbar_bounds, cbar_cmap.N)

    cbar = fig.colorbar(
        cm.ScalarMappable(cmap=cbar_cmap, norm=cbar_norm),
        ax=ax,
        orientation="vertical",
        extend="max",
        extendfrac="auto",
        extendrect=True,
        spacing="proportional",
    )
    tick_step = np.floor(maxpeaks / 5)  # 5 ticks for cbar
    cbar_ticks = np.append(np.arange(1, maxpeaks - 1, tick_step), maxpeaks + 1)
    cbar.set_ticks(cbar_ticks + 0.5)
    cbar.set_ticklabels(cbar_ticks - 1)
    cbar.ax.tick_params(labelsize=5)
    cbar.set_label("Number of Peaks", fontsize=7)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # remove padding
    if savepath is not None:
        fig.savefig(savepath, dpi=300, transparent=True)
    return fig
