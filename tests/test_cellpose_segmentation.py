import numpy as np
import pytest
import warnings
from unittest.mock import patch, MagicMock

from pykrait.preprocessing.segmentation import create_cellpose_segmentation


def test_invalid_model_path_raises_file_not_found(tmp_path):
    image = np.random.rand(128, 128)
    invalid_path = tmp_path / "nonexistent_model.pth"

    with pytest.raises(FileNotFoundError):
        create_cellpose_segmentation(image, str(invalid_path))


def test_invalid_image_shape_raises_value_error():
    bad_image = np.random.rand(2, 3, 4, 5)  # Too many dimensions

    with pytest.raises(ValueError):
        create_cellpose_segmentation(bad_image, "cpsam")


@patch("cellpose.models.CellposeModel")
def test_model_fallback_to_cpu(mock_cellpose_model):
    # Simulate GPU failure and successful CPU fallback
    mock_cellpose_model.side_effect = [
        Exception("GPU error"),
        MagicMock(eval=lambda x: (np.ones((128, 128), dtype=int), None, None)),
    ]

    image = np.random.rand(128, 128)
    result = create_cellpose_segmentation(image, "cpsam")

    assert isinstance(result, np.ndarray)
    assert result.shape == (128, 128)
    assert np.max(result) == 1


@patch("cellpose.models.CellposeModel")
def test_mask_all_zeros_triggers_warning(mock_cellpose_model):
    mock_model = MagicMock()
    mock_model.eval.return_value = (np.zeros((128, 128), dtype=int), None, None)
    mock_cellpose_model.return_value = mock_model

    image = np.random.rand(128, 128)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        mask = create_cellpose_segmentation(image, "cpsam")
        assert len(w) == 1
        assert "only zeros" in str(w[0].message)

    assert isinstance(mask, np.ndarray)
    assert np.all(mask == 0)


@patch("cellpose.models.CellposeModel")
def test_valid_mask_output(mock_cellpose_model):
    fake_mask = np.random.randint(0, 4, size=(128, 128))
    mock_model = MagicMock()
    mock_model.eval.return_value = (fake_mask, None, None)
    mock_cellpose_model.return_value = mock_model

    image = np.random.rand(128, 128)
    mask = create_cellpose_segmentation(image, "cpsam")

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (128, 128)
    assert np.array_equal(mask, fake_mask)


# TODO: run test with real data
