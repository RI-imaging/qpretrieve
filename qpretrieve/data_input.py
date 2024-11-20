import numpy as np


def check_data_input(data):
    """Figure out what data input is provided."""
    if len(data.shape) == 3:
        if data.shape[-1] in [1, 3, 4]:
            # take the first slice (we have alpha or RGB information)
            data = data[:, :, 0]
        else:
            # we have a 3D image stack (z, y, x)
            pass
    if len(data.shape) == 2:
        # we have a 2D image (y, x). convert to (z, y, z)
        data = data[np.newaxis, :, :]

    return data.copy()
