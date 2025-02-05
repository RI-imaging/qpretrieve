import numpy as np


def _mean_2d(data):
    """Exists for testing against mean_3d"""
    data -= data.mean()
    return data


def mean_3d(data: np.ndarray) -> np.ndarray:
    """
    Subtract mean inplace from 3D array `data` for every
    2D array along z axis.
    """
    # The mean array here is (1000,), so we need to add newaxes for subtraction
    # (1000, 5, 5) -= (1000, 1, 1)
    data -= data.mean(axis=(-2, -1))[:, np.newaxis, np.newaxis]
    return data


def _padding_2d(data, order, dtype):
    """Exists for testing against padding_3d"""
    # this is faster than np.pad
    datapad = np.zeros((order, order), dtype=dtype)
    # we could of course use np.atleast_3d here
    datapad[:data.shape[0], :data.shape[1]] = data
    return datapad


def padding_3d(data: np.ndarray, size: int, dtype: np.dtype) -> np.ndarray:
    """Pad a 3D array in the second and third dimensions (y, x) to `size`

    Parameters
    ----------
    data
        3d array. The padding will be applied to the axes (y,x) only.
    size
        The data will be padded to this size in the (y, x) dimensions.
    dtype
        data type of the padded array.

    """
    z, y, x = data.shape
    # this is faster than np.pad
    datapad = np.zeros((z, size, size), dtype=dtype)
    datapad[:, :y, :x] = data
    return datapad
