"""These are tests from qpimage for Cupy imported `FFTFilter`s"""
import numpy as np

import qpretrieve
from qpretrieve.fourier import FFTFilterCupy3D, FFTFilterCupy, FFTFilterNumpy


def test_get_field_cupy3d(hologram):
    data1 = hologram
    data_rp = np.array([data1, data1, data1, data1, data1])

    holo1 = qpretrieve.OffAxisHologram(data_rp,
                                       fft_interface=FFTFilterCupy3D,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res1 = holo1.run_pipeline(**kwargs)
    assert res1.shape == (5, 64, 64)

    holo1 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterNumpy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res2 = holo1.run_pipeline(**kwargs)
    assert res2.shape == (64, 64)

    assert not np.all(res1[0] == res2)

    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(3, 1)
    # ax1, ax2, ax3 = axes
    # ax1.imshow(np.abs(res1[0]))
    # ax2.imshow(np.abs(res2))
    # ax3.imshow(np.abs(res2)-np.abs(res1[0]))
    # plt.show()


def test_get_field_compare_FFTFilters(hologram):
    data1 = hologram

    holo1 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterNumpy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res1 = holo1.run_pipeline(**kwargs)
    assert res1.shape == (64, 64)

    holo1 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterCupy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res2 = holo1.run_pipeline(**kwargs)
    assert res2.shape == (64, 64)

    assert not np.all(res1 == res2)
