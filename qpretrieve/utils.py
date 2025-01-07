import numpy as np


def mean_2d(data):
    data -= data.mean()
    return data


def mean_3d(data):
    # calculate mean of the images along the z-axis.
    # The mean array here is (1000,), so we need to add newaxes for subtraction
    # (1000, 5, 5) -= (1000, 1, 1)
    data -= data.mean(axis=(-2, -1))[:, np.newaxis, np.newaxis]
    return data


def padding_2d(data, order, dtype):
    # this is faster than np.pad
    datapad = np.zeros((order, order), dtype=dtype)
    # we could of course use np.atleast_3d here
    datapad[:data.shape[0], :data.shape[1]] = data
    return datapad


def padding_3d(data, order, dtype):
    z, y, x = data.shape
    # this is faster than np.pad
    datapad = np.zeros((z, order, order), dtype=dtype)
    datapad[:, :y, :x] = data
    return datapad
