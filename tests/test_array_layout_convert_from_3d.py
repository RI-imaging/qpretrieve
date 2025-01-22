import numpy as np
import pytest

from qpretrieve.data_array_layout import (
    convert_3d_data_to_array_layout,
    _convert_3d_to_2d, _convert_3d_to_rgba, _convert_3d_to_rgb,
)


def test_convert_3d_data_to_2d():
    data = np.zeros(shape=(10, 256, 256))
    array_layout = "2d"

    data_new = convert_3d_data_to_array_layout(data, array_layout)
    data_direct = _convert_3d_to_2d(data)  # this is the internal function

    assert np.array_equal(data[0], data_new)
    assert data_new.shape == data_direct.shape == (256, 256)
    assert np.array_equal(data_direct, data_new)


def test_convert_3d_data_to_rgb():
    data = np.zeros(shape=(10, 256, 256))
    array_layout = "rgb"

    data_new = convert_3d_data_to_array_layout(data, array_layout)
    data_direct = _convert_3d_to_rgb(data)  # this is the internal function

    assert data_new.shape == data_direct.shape == (256, 256, 3)
    assert np.array_equal(data_direct, data_new)


def test_convert_3d_data_to_rgba():
    data = np.zeros(shape=(10, 256, 256))
    array_layout = "rgba"

    data_new = convert_3d_data_to_array_layout(data, array_layout)
    data_direct = _convert_3d_to_rgba(data)  # this is the internal function

    assert data_new.shape == data_direct.shape == (256, 256, 4)
    assert np.array_equal(data_direct, data_new)


def test_convert_3d_data_to_array_layout_bad_input():
    data = np.zeros(shape=(10, 256, 256))
    array_layout = "5d"

    with pytest.raises(AssertionError, match="`array_layout` not allowed."):
        convert_3d_data_to_array_layout(data, array_layout)

    data = np.zeros(shape=(256, 256))
    array_layout = "2d"

    with pytest.raises(AssertionError, match="The data should be 3d"):
        convert_3d_data_to_array_layout(data, array_layout)
