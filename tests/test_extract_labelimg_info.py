import numpy as np
import pandas as pd
import pytest
from skimage.draw import disk
from skimage.measure import label
from scipy.sparse.csgraph import floyd_warshall
from pykrait.preprocessing.timeseries_extraction import (
    extract_cell_properties,
    get_adjacency_matrix,
)


@pytest.fixture
def sample_mask():
    """
    Create a sample mask with two labeled circular regions.
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    rr1, cc1 = disk((30, 30), 10)
    rr2, cc2 = disk((70, 70), 15)
    img[rr1, cc1] = 1
    img[rr2, cc2] = 2
    labeled = label(img)
    return labeled


def test_extract_cell_properties_shape(sample_mask):
    props, positions = extract_cell_properties(sample_mask)

    assert isinstance(props, pd.DataFrame), "Output properties should be a DataFrame"
    assert isinstance(positions, np.ndarray), "Positions should be a numpy array"
    assert props.shape[0] == sample_mask.max(), (
        "Number of properties rows should match number of labels"
    )
    assert positions.shape == (sample_mask.max(), 2), (
        "Positions should have shape (n_cells, 2)"
    )


def test_extract_cell_properties_columns(sample_mask):
    props, _ = extract_cell_properties(sample_mask)

    expected_columns = {
        "label",
        "centroid,0",
        "centroid,1",
        "intensity_max",
        "area",
        "axis_major_length",
        "axis_minor_length",
        "perimeter",
    }
    assert expected_columns.issubset(set(props.columns)), (
        "Missing expected columns in properties DataFrame"
    )


def test_extract_cell_properties_consistency(sample_mask):
    props, positions = extract_cell_properties(sample_mask)

    # Check that the position matches centroid rounded to uint16
    for i in range(sample_mask.max()):
        centroid_x = int(round(props.loc[i, "centroid,1"]))
        centroid_y = int(round(props.loc[i, "centroid,0"]))
        assert positions[i, 0] == centroid_x
        assert positions[i, 1] == centroid_y


def test_extract_cell_properties_label_indexing(sample_mask):
    props, _ = extract_cell_properties(sample_mask)

    expected_labels = set(range(sample_mask.max()))
    actual_labels = set(props["label"].astype(int).tolist())

    assert expected_labels == actual_labels, (
        "Labels should be 0-indexed and cover all regions"
    )


# Tests for adjacency matrix extraction
@pytest.fixture
def simple_mask():
    """
    Create a small labeled mask with three separate regions:
    Region 1: Top-left
    Region 2: Top-right
    Region 3: Bottom-center
    """
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[1:3, 7:9] = 2
    mask[7:9, 4:6] = 3
    return mask


def test_adjacency_matrix_shape(simple_mask):
    adj_matrix, sp_matrix = get_adjacency_matrix(simple_mask, neighbour_tolerance=3)
    num_cells = simple_mask.max()

    assert adj_matrix.shape == (num_cells, num_cells), (
        "Adjacency matrix should be (N, N)"
    )
    assert sp_matrix.shape == (num_cells, num_cells), (
        "Shortest path matrix should be (N, N)"
    )


def test_adjacency_matrix_symmetry(simple_mask):
    adj_matrix, _ = get_adjacency_matrix(simple_mask, neighbour_tolerance=3)

    assert np.array_equal(adj_matrix, adj_matrix.T), (
        "Adjacency matrix should be symmetric"
    )


def test_adjacency_self_zero(simple_mask):
    adj_matrix, _ = get_adjacency_matrix(simple_mask, neighbour_tolerance=2)
    assert np.all(np.diag(adj_matrix) == 0), "No node should be adjacent to itself"


def test_shortest_path_with_floyd(simple_mask):
    adj_matrix, sp_matrix = get_adjacency_matrix(simple_mask, neighbour_tolerance=3)

    expected_sp = np.triu(floyd_warshall(adj_matrix), k=1)
    np.testing.assert_array_equal(
        sp_matrix,
        expected_sp,
        "Shortest path matrix should match Floyd-Warshall result",
    )


def test_adjacency_within_tolerance(simple_mask):
    # Set a small enough tolerance so none are adjacent
    adj_matrix, _ = get_adjacency_matrix(simple_mask, neighbour_tolerance=1)
    assert not adj_matrix.any(), (
        "With zero/low tolerance, no adjacency should be detected"
    )

    # Set a large tolerance to ensure at least some adjacency
    adj_matrix, _ = get_adjacency_matrix(simple_mask, neighbour_tolerance=100)
    # With loose enough tolerance, all cells should be connected
    assert (adj_matrix.sum(axis=0) > 0).all(), (
        "All cells should have at least one neighbor with high tolerance"
    )


def test_edge_case_background_only():
    # Mask with only background
    mask = np.zeros((5, 5), dtype=np.uint8)
    adj_matrix, sp_matrix = get_adjacency_matrix(mask, neighbour_tolerance=3)
    assert adj_matrix.shape == (0, 0), "Should return empty matrices if no labels"
    assert sp_matrix.shape == (0, 0), "Should return empty matrices if no labels"


def test_adjacency_matrix_verbose():
    mask = np.zeros((5, 5), dtype=np.uint8)
    _, _ = get_adjacency_matrix(mask, neighbour_tolerance=1, verbose=True)
    _, _ = get_adjacency_matrix(mask, neighbour_tolerance=1, verbose=False)
