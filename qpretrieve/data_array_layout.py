"""
Module that provides convenience functions for converting data between
array layouts.
"""

import numpy as np


def get_allowed_array_layouts() -> list:
    return [
        "rgb",
        "rgba",
        "3d",
        "2d",
    ]


def convert_data_to_3d_array_layout(
        data: np.ndarray) -> tuple[np.ndarray, str]:
    """Convert the data to the 3d array_layout

    Returns
    -------
    data_out
        3d version of the data
    array_layout
        original array layout for future reference

    Notes
    -----
    If input is either a RGB or RGBA array layout as input, the first
    channel is taken as the image to process. In other words, it is assumed
    that all channels contain the same information, so the first channel is
    used. 3D RGB/RGBA array layouts, such as (50, 256, 256, 3), are not allowed
    (yet).

    """
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
    return data.copy(), array_layout


def convert_3d_data_to_array_layout(
        data: np.ndarray, array_layout: str) -> np.ndarray:
    """Convert the 3d data to the desired `array_layout`.

    Returns
    -------
    data_out : np.ndarray
        input `data` with the given `array layout`

    Notes
    -----
    Currently, this function is limited to converting from 3d to other
    array layouts. Perhaps if there is demand in the future,
    this can be generalised for other conversions.

    """
    assert array_layout in get_allowed_array_layouts(), (
        f"`array_layout` not allowed. "
        f"Allowed layouts are: {get_allowed_array_layouts()}.")
    assert len(data.shape) == 3, (
        f"The data should be 3d, got {len(data.shape)=}")
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


def _convert_rgb_to_3d(data_input: np.ndarray) -> tuple[np.ndarray, str]:
    data = data_input[:, :, 0]
    data = data[np.newaxis, :, :]
    array_layout = "rgb"
    return data, array_layout


def _convert_rgba_to_3d(data_input: np.ndarray) -> tuple[np.ndarray, str]:
    data, _ = _convert_rgb_to_3d(data_input)
    array_layout = "rgba"
    return data, array_layout


def _convert_2d_to_3d(data_input: np.ndarray) -> tuple[np.ndarray, str]:
    data = data_input[np.newaxis, :, :]
    array_layout = "2d"
    return data, array_layout


def _convert_3d_to_rgb(data_input: np.ndarray) -> np.ndarray:
    data = data_input[0]
    data = np.dstack((data, data, data))
    return data


def _convert_3d_to_rgba(data_input: np.ndarray) -> np.ndarray:
    data = data_input[0]
    data = np.dstack((data, data, data, np.ones_like(data)))
    return data


def _convert_3d_to_2d(data_input: np.ndarray) -> np.ndarray:
    return data_input[0]
