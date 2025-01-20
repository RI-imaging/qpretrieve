import numpy as np
import warnings

allowed_orig_data_fmts = [
    "rgb",
    "rgba",
    "3d",
    "2d",
]


def check_data_input_format(data_input):
    """Figure out what data input is provided."""
    if len(data_input.shape) == 3:
        if data_input.shape[-1] in [1, 2, 3]:
            # take the first slice (we have alpha or RGB information)
            data, orig_data_fmt = _convert_rgb_to_3d(data_input)
        elif data_input.shape[-1] == 4:
            # take the first slice (we have alpha or RGB information)
            data, orig_data_fmt = _convert_rgba_to_3d(data_input)
        else:
            # we have a 3D image stack (z, y, x)
            data, orig_data_fmt = data_input, "3d"
    elif len(data_input.shape) == 2:
        # we have a 2D image (y, x). convert to (z, y, z)
        data, orig_data_fmt = _convert_2d_to_3d(data_input)
    else:
        raise ValueError(f"data_input shape must be 2d or 3d, "
                         f"got shape {data_input.shape}.")
    return data.copy(), orig_data_fmt


def revert_to_data_input_format(orig_data_fmt, field):
    """Convert the outputted field shape to the original input shape,
    for user convenience."""
    assert orig_data_fmt in allowed_orig_data_fmts
    assert len(field.shape) == 3, "the field should be 3d"
    field = field.copy()
    if orig_data_fmt == "rgb":
        field = _revert_3d_to_rgb(field)
    elif orig_data_fmt == "rgba":
        field = _revert_3d_to_rgba(field)
    elif orig_data_fmt == "3d":
        field = field
    else:
        field = _revert_3d_to_2d(field)
    return field


def _convert_rgb_to_3d(data_input):
    data = data_input[:, :, 0]
    data = data[np.newaxis, :, :]
    orig_data_fmt = "rgb"
    warnings.warn(f"Format of input data detected as {orig_data_fmt}. "
                  f"The first channel will be used for processing")
    return data, orig_data_fmt


def _convert_rgba_to_3d(data_input):
    data, _ = _convert_rgb_to_3d(data_input)
    orig_data_fmt = "rgba"
    return data, orig_data_fmt


def _convert_2d_to_3d(data_input):
    data = data_input[np.newaxis, :, :]
    orig_data_fmt = "2d"
    return data, orig_data_fmt


def _revert_3d_to_rgb(data_input):
    data = data_input[0]
    data = np.dstack((data, data, data))
    return data


def _revert_3d_to_rgba(data_input):
    data = data_input[0]
    data = np.dstack((data, data, data, np.ones_like(data)))
    return data


def _revert_3d_to_2d(data_input):
    return data_input[0]
