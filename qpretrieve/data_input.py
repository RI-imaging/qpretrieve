import numpy as np

allowed_data_formats = [
    "rgb",
    "rgba",
    "3d",
    "2d",
]


def check_data_input_form(data_input):
    """Figure out what data input is provided."""
    if len(data_input.shape) == 3:
        if data_input.shape[-1] in [1, 2, 3]:
            # take the first slice (we have alpha or RGB information)
            data, data_format = _convert_rgb_to_3d(data_input)
        elif data_input.shape[-1] == 4:
            # take the first slice (we have alpha or RGB information)
            data, data_format = _convert_rgba_to_3d(data_input)
        else:
            # we have a 3D image stack (z, y, x)
            data, data_format = data_input, "3d"
    elif len(data_input.shape) == 2:
        # we have a 2D image (y, x). convert to (z, y, z)
        data, data_format = _convert_2d_to_3d(data_input)
    else:
        raise ValueError(f"data_input shape must be 2d or 3d, "
                         f"got shape {data_input.shape}.")
    return data.copy(), data_format


def revert_to_data_input_shape(data_format, field):
    """Convert the outputted field shape to the original input shape,
    for user convenience."""
    assert data_format in allowed_data_formats
    assert len(field.shape) == 3, "the field should be 3d"
    field = field.copy()
    if data_format == "rgb":
        field = _convert_3d_to_rgb(field)
    elif data_format == "rgba":
        field = _convert_3d_to_rgba(field)
    elif data_format == "3d":
        field = field
    else:
        field = _convert_3d_to_2d(field)
    return field


def _convert_rgb_to_3d(data_input):
    data = data_input[:, :, 0]
    data = data[np.newaxis, :, :]
    data_format = "rgb"
    return data, data_format


def _convert_rgba_to_3d(data_input):
    data, _ = _convert_rgb_to_3d(data_input)
    data_format = "rgba"
    return data, data_format


def _convert_2d_to_3d(data_input):
    data = data_input[np.newaxis, :, :]
    data_format = "2d"
    return data, data_format


def _convert_3d_to_rgb(field):
    field = field[0]
    field = np.dstack((field, field, field))
    return field


def _convert_3d_to_rgba(field):
    field = field[0]
    field = np.dstack((field, field, field, np.ones_like(field)))
    return field


def _convert_3d_to_2d(field):
    return field[0]
