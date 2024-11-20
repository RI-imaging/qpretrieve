import numpy as np

from qpretrieve.data_input import check_data_input


def test_check_data_input_2d():
    data = np.zeros(shape=(256, 256))

    data_new = check_data_input(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data)


def test_check_data_input_3d_image_stack():
    data = np.zeros(shape=(50, 256, 256))

    data_new = check_data_input(data)

    assert data_new.shape == (50, 256, 256)
    assert np.array_equal(data_new, data)


def test_check_data_input_3d_rgb():
    data = np.zeros(shape=(256, 256, 3))

    data_new = check_data_input(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data[:, :, 0])


def test_check_data_input_3d_rgba():
    data = np.zeros(shape=(256, 256, 4))

    data_new = check_data_input(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data[:, :, 0])
