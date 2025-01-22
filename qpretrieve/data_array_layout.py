"""
Module that provides convenience functions for converting data between
array layouts.
"""

import numpy as np
import warnings


def get_allowed_array_layouts() -> list:
    return [
        "rgb",
        "rgba",
        "3d",
        "2d",
    ]


def convert_data_to_3d_array_layout(data):
    """Convert the data to the 3d array_layout."""
    if len(data.shape) == 3:
        if data.shape[-1] in [1, 2, 3]:
            # take the first slice (we have alpha or RGB information)
            data, array_layout = _convert_rgb_to_3d(data)
        elif data.shape[-1] == 4:
            # take the first slice (we have alpha or RGB information)
            data, array_layout = _convert_rgba_to_3d(data)
        else:
            # we have a 3D image stack (z, y, x)
            data, array_layout = data, "3d"
    elif len(data.shape) == 2:
        # we have a 2D image (y, x). convert to (z, y, z)
        data, array_layout = _convert_2d_to_3d(data)
    else:
        raise ValueError(f"data_input shape must be 2d or 3d, "
                         f"got shape {data.shape}.")
    warnings.warn(f"Format of input data was detected as '{array_layout}'. "
                  f"The new output data format is '3d'. To get your data in "
                  f"the original format use, for example, "
                  f"`oah.get_array_with_input_layout(data)`.")
    return data.copy(), array_layout


def convert_3d_data_to_array_layout(data, array_layout):
    """Convert the 3d data to the desired `array_layout`"""
    assert array_layout in get_allowed_array_layouts()
    assert len(data.shape) == 3, "the data should be 3d"
    data = data.copy()
    if array_layout == "rgb":
        data = _convert_3d_to_rgb(data)
    elif array_layout == "rgba":
        data = _convert_3d_to_rgba(data)
    elif array_layout == "3d":
        data = data
    else:
        data = _convert_3d_to_2d(data)
    return data


def _convert_rgb_to_3d(data_input):
    data = data_input[:, :, 0]
    data = data[np.newaxis, :, :]
    array_layout = "rgb"
    warnings.warn(f"Format of input data detected as {array_layout}. "
                  f"The first channel will be used for processing")
    return data, array_layout


def _convert_rgba_to_3d(data_input):
    data, _ = _convert_rgb_to_3d(data_input)
    array_layout = "rgba"
    return data, array_layout


def _convert_2d_to_3d(data_input):
    data = data_input[np.newaxis, :, :]
    array_layout = "2d"
    return data, array_layout


def _convert_3d_to_rgb(data_input):
    data = data_input[0]
    data = np.dstack((data, data, data))
    return data


def _convert_3d_to_rgba(data_input):
    data = data_input[0]
    data = np.dstack((data, data, data, np.ones_like(data)))
    return data


def _convert_3d_to_2d(data_input):
    return data_input[0]
