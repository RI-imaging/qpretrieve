import numpy as np

from qpretrieve.data_input import check_data_input_form


def test_check_data_input_2d():
    data = np.zeros(shape=(256, 256))

    data_new, data_format = check_data_input_form(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data)
    assert data_format == "2d"


def test_check_data_input_3d_image_stack():
    data = np.zeros(shape=(50, 256, 256))

    data_new, data_format = check_data_input_form(data)

    assert data_new.shape == (50, 256, 256)
    assert np.array_equal(data_new, data)
    assert data_format == "3d"


def test_check_data_input_3d_rgb():
    data = np.zeros(shape=(256, 256, 3))

    data_new, data_format = check_data_input_form(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data[:, :, 0])
    assert data_format == "rgb"


def test_check_data_input_3d_rgba():
    data = np.zeros(shape=(256, 256, 4))

    data_new, data_format = check_data_input_form(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data[:, :, 0])
    assert data_format == "rgba"
