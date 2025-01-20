import numpy as np
import pytest

from qpretrieve.data_input import check_data_input_format


def test_check_data_input_2d():
    data = np.zeros(shape=(256, 256))

    data_new, orig_data_fmt = check_data_input_format(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data)
    assert orig_data_fmt == "2d"


def test_check_data_input_3d_image_stack():
    data = np.zeros(shape=(50, 256, 256))

    data_new, orig_data_fmt = check_data_input_format(data)

    assert data_new.shape == (50, 256, 256)
    assert np.array_equal(data_new, data)
    assert orig_data_fmt == "3d"


def test_check_data_input_3d_rgb():
    data = np.zeros(shape=(256, 256, 3))

    with pytest.warns(UserWarning):
        data_new, orig_data_fmt = check_data_input_format(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data[:, :, 0])
    assert orig_data_fmt == "rgb"


def test_check_data_input_3d_rgba():
    data = np.zeros(shape=(256, 256, 4))

    with pytest.warns(UserWarning):
        data_new, orig_data_fmt = check_data_input_format(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data[:, :, 0])
    assert orig_data_fmt == "rgba"
