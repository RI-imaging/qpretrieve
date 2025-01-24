import numpy as np

from qpretrieve.utils import _padding_2d, padding_3d, _mean_2d, mean_3d


def test_mean_subtraction():
    data_3d = np.random.rand(1000, 5, 5).astype(np.float32)
    ind = 5
    data_2d = data_3d.copy()[ind]

    data_2d = _mean_2d(data_2d)
    data_3d = mean_3d(data_3d)

    assert np.array_equal(data_3d[ind], data_2d)


def test_mean_subtraction_consistent_2d_3d():
    """Probably a bit too cumbersome, and changes the default 2d pipeline."""
    data_3d = np.random.rand(1000, 5, 5).astype(np.float32)
    ind = 5
    data_2d = data_3d.copy()[ind]

    # too cumbersome
    data_2d = np.atleast_3d(data_2d)
    data_2d = np.swapaxes(np.swapaxes(data_2d, 0, 2), 1, 2)
    data_2d -= data_2d.mean(axis=(-2, -1))[:, np.newaxis, np.newaxis]

    data_3d = np.atleast_3d(data_3d.copy())
    data_3d -= data_3d.mean(axis=(-2, -1))[:, np.newaxis, np.newaxis]

    assert np.array_equal(data_3d[ind], data_2d[0])


def test_batch_padding():
    data_3d = np.random.rand(1000, 100, 320).astype(np.float32)
    ind = 5
    data_2d = data_3d.copy()[ind]
    order = 512
    dtype = float

    data_2d_padded = _padding_2d(data_2d, order, dtype)
    data_3d_padded = padding_3d(data_3d, order, dtype)

    assert np.array_equal(data_3d_padded[ind], data_2d_padded)
