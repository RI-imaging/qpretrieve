"""These are tests from qpimage for Cupy imported `FFTFilter`s"""
import numpy as np

import qpretrieve
from qpretrieve.fourier import FFTFilterCupy3D, FFTFilterCupy, FFTFilterNumpy


def test_get_field_compare_FFTFilters(hologram):
    data1 = hologram

    holo1 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterNumpy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res1 = holo1.run_pipeline(**kwargs)

    holo1 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterCupy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res2 = holo1.run_pipeline(**kwargs)

    assert res1.shape == (64, 64)
    assert res2.shape == (64, 64)
    assert not np.all(res1 == res2)
    assert np.allclose(res1, res2)


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


def test_get_field_cupy3d_scale_to_filter(hologram):
    data1 = hologram
    data_rp = np.array([data1, data1, data1, data1, data1])

    holo1 = qpretrieve.OffAxisHologram(data_rp,
                                       fft_interface=FFTFilterCupy3D,
                                       padding=True)
    kwargs = dict(filter_name="disk", filter_size=1 / 3,
                  scale_to_filter=True)
    res1 = holo1.run_pipeline(**kwargs)

    holo2 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterNumpy,
                                       padding=True)
    kwargs = dict(filter_name="disk", filter_size=1 / 3,
                  scale_to_filter=True)
    res2 = holo2.run_pipeline(**kwargs)

    assert res1.shape == (5, 18, 18)
    assert res2.shape == (18, 18)
    assert np.allclose(res1[0], res2)
