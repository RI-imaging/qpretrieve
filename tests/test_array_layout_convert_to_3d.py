import numpy as np

from qpretrieve.data_array_layout import convert_data_to_3d_array_layout


def test_check_data_input_2d():
    data = np.zeros(shape=(256, 256))

    data_new, orig_array_layout = convert_data_to_3d_array_layout(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data)
    assert orig_array_layout == "2d"


def test_check_data_input_3d_image_stack():
    data = np.zeros(shape=(50, 256, 256))

    data_new, orig_array_layout = convert_data_to_3d_array_layout(data)

    assert data_new.shape == (50, 256, 256)
    assert np.array_equal(data_new, data)
    assert orig_array_layout == "3d"


def test_check_data_input_3d_rgb():
    data = np.zeros(shape=(256, 256, 3))

    data_new, orig_array_layout = convert_data_to_3d_array_layout(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data[:, :, 0])
    assert orig_array_layout == "rgb"


def test_check_data_input_3d_rgba():
    data = np.zeros(shape=(256, 256, 4))

    data_new, orig_array_layout = convert_data_to_3d_array_layout(data)

    assert data_new.shape == (1, 256, 256)
    assert np.array_equal(data_new[0], data[:, :, 0])
    assert orig_array_layout == "rgba"
