import numpy as np


def test_mean_subtraction():
    data_3D = np.random.rand(1000, 5, 5).astype(np.float32)
    ind = 5
    data_2D = data_3D.copy()[ind]

    data_2D -= data_2D.mean()
    # calculate mean of the images along the z-axis.
    # The mean array here is (1000,), so we need to add newaxes for subtraction
    # (1000, 5, 5) -= (1000, 1, 1)
    data_3D -= data_3D.mean(axis=(-2, -1))[:, np.newaxis, np.newaxis]

    assert np.array_equal(data_3D[ind], data_2D)


def test_mean_subtraction_consistent_2D_3D():
    """Probably a bit too cumbersome, and changes the default 2D pipeline."""
    data_3D = np.random.rand(1000, 5, 5).astype(np.float32)
    ind = 5
    data_2D = data_3D.copy()[ind]

    # too cumbersome
    data_2D = np.atleast_3d(data_2D)
    data_2D = np.swapaxes(np.swapaxes(data_2D, 0, 2), 1, 2)
    data_2D -= data_2D.mean(axis=(-2, -1))[:, np.newaxis, np.newaxis]

    data_3D = np.atleast_3d(data_3D.copy())
    data_3D -= data_3D.mean(axis=(-2, -1))[:, np.newaxis, np.newaxis]

    assert np.array_equal(data_3D[ind], data_2D[0])
