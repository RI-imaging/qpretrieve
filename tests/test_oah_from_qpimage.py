"""These are tests from qpimage"""
import numpy as np
import pytest

import qpretrieve
from qpretrieve.interfere import if_oah
from qpretrieve.fourier import FFTFilterNumpy, FFTFilterScipy, FFTFilterPyFFTW


def test_find_sideband():
    size = 40
    ft_data = np.zeros((size, size))
    fx = np.fft.fftshift(np.fft.fftfreq(size))
    ft_data[2, 3] = 1
    ft_data[-3, -2] = 1

    sb1 = if_oah.find_peak_cosine(ft_data=ft_data)
    assert np.allclose(sb1, (fx[2], fx[3]))


def test_fourier2dpad():
    data = np.zeros((100, 120))
    fft1 = qpretrieve.fourier.FFTFilterNumpy(data)
    assert fft1.shape == (256, 256)

    fft2 = qpretrieve.fourier.FFTFilterNumpy(data, padding=False)
    assert fft2.shape == data.shape


def test_get_field_error_bad_filter_size(hologram):
    data = hologram

    holo = qpretrieve.OffAxisHologram(data)
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        holo.run_pipeline(filter_size=2)


def test_get_field_error_bad_filter_size_interpretation_frequency_index(
        hologram):
    data = hologram
    holo = qpretrieve.OffAxisHologram(data)

    with pytest.raises(ValueError,
                       match=r"must be between 0 and 64"):
        holo.run_pipeline(filter_size_interpretation="frequency index",
                          filter_size=64)


def test_get_field_error_invalid_interpretation(hologram):
    data = hologram
    holo = qpretrieve.OffAxisHologram(data)

    with pytest.raises(ValueError,
                       match="Invalid value for `filter_size_interpretation`"):
        holo.run_pipeline(filter_size_interpretation="blequency")


def test_get_field_filter_names(hologram):
    data = hologram
    holo = qpretrieve.OffAxisHologram(data)

    kwargs = dict(sideband=+1,
                  filter_size=1 / 3)

    r_disk = holo.run_pipeline(filter_name="disk", **kwargs)
    assert np.allclose(
        r_disk[32, 32], 97.307780444912936 - 76.397860381241372j)

    r_smooth_disk = holo.run_pipeline(filter_name="smooth disk", **kwargs)
    assert np.allclose(r_smooth_disk[32, 32],
                       108.36438759594623 - 67.1806221692573j)

    r_gauss = holo.run_pipeline(filter_name="gauss", **kwargs)
    assert np.allclose(r_gauss[32, 32],
                       108.2914187451138 - 67.1823527237741j)

    r_square = holo.run_pipeline(filter_name="square", **kwargs)
    assert np.allclose(
        r_square[32, 32], 102.3285348843612 - 74.139058665601155j)

    r_smsquare = holo.run_pipeline(filter_name="smooth square", **kwargs)
    assert np.allclose(
        r_smsquare[32, 32], 108.36651862466393 - 67.17988960794392j)

    r_tukey = holo.run_pipeline(filter_name="tukey", **kwargs)
    assert np.allclose(
        r_tukey[32, 32], 113.4826495540899 - 59.546232775481869j)

    try:
        holo.run_pipeline(filter_name="unknown", **kwargs)
    except ValueError:
        pass
    else:
        assert False, "unknown filter accepted"


@pytest.mark.parametrize("hologram", [62, 63, 64], indirect=["hologram"])
def test_get_field_interpretation_fourier_index(hologram):
    """Filter size in Fourier space using Fourier index new in 0.7.0"""
    data = hologram
    holo = qpretrieve.OffAxisHologram(data)

    ft_data = holo.fft_origin
    holo.run_pipeline()
    fsx, fsy = holo.pipeline_kws["sideband_freq"]

    kwargs1 = dict(filter_name="disk",
                   filter_size=1 / 3,
                   filter_size_interpretation="sideband distance")
    res1 = holo.run_pipeline(**kwargs1)

    filter_size_fi = np.sqrt(fsx ** 2 + fsy ** 2) / 3 * ft_data.shape[0]
    kwargs2 = dict(filter_name="disk",
                   filter_size=filter_size_fi,
                   filter_size_interpretation="frequency index",
                   )
    res2 = holo.run_pipeline(**kwargs2)
    assert np.all(res1 == res2)


@pytest.mark.parametrize("hologram", [62, 63, 64], indirect=["hologram"])
def test_get_field_interpretation_fourier_index_control(hologram):
    """Filter size in Fourier space using Fourier index new in 0.7.0"""
    data = hologram
    holo = qpretrieve.OffAxisHologram(data)

    ft_data = holo.fft_origin
    holo.run_pipeline()
    fsx, fsy = holo.pipeline_kws["sideband_freq"]

    evil_factor = 1.1

    kwargs1 = dict(filter_name="disk",
                   filter_size=1 / 3 * evil_factor,
                   filter_size_interpretation="sideband distance"
                   )
    res1 = holo.run_pipeline(**kwargs1)

    filter_size_fi = np.sqrt(fsx ** 2 + fsy ** 2) / 3 * ft_data.shape[0]
    kwargs2 = dict(filter_name="disk",
                   filter_size=filter_size_fi,
                   filter_size_interpretation="frequency index",
                   )
    res2 = holo.run_pipeline(**kwargs2)
    assert not np.all(res1 == res2)


@pytest.mark.parametrize("hologram", [62, 63, 64, 134, 135],
                         indirect=["hologram"])
@pytest.mark.parametrize("filter_size", [17, 17.01])
def test_get_field_interpretation_fourier_index_mask_1(hologram, filter_size):
    """Make sure filter size in Fourier space pixels is correct"""
    data = hologram
    holo = qpretrieve.OffAxisHologram(data)

    kwargs2 = dict(filter_name="disk",
                   filter_size=filter_size,
                   filter_size_interpretation="frequency index",
                   )
    holo.run_pipeline(**kwargs2)
    mask = holo.fft_filtered.real != 0

    # We get 17*2+1, because we measure from the center of Fourier
    # space and a pixel is included if its center is withing the
    # perimeter of the disk.
    assert np.sum(np.sum(mask, axis=0) != 0) == 17 * 2 + 1


@pytest.mark.parametrize("hologram", [62, 63, 64, 134, 135],
                         indirect=["hologram"])
def test_get_field_interpretation_fourier_index_mask_2(hologram):
    """Filter size in Fourier space using Fourier index new in 0.7.0"""
    data = hologram
    holo = qpretrieve.OffAxisHologram(data)

    kwargs2 = dict(filter_name="disk",
                   filter_size=16.99,
                   filter_size_interpretation="frequency index"
                   )
    holo.run_pipeline(**kwargs2)
    mask = holo.fft_filtered.real != 0

    # We get two points less than in the previous test, because we
    # loose on each side of the spectrum.
    assert np.sum(np.sum(mask, axis=0) != 0) == 17 * 2 - 1


def test_get_field_int_copy(hologram):
    data = hologram
    data = np.array(data, dtype=int)

    kwargs = dict(filter_size=1 / 3)

    holo1 = qpretrieve.OffAxisHologram(data, copy=False)
    res1 = holo1.run_pipeline(**kwargs)

    holo2 = qpretrieve.OffAxisHologram(data, copy=True)
    res2 = holo2.run_pipeline(**kwargs)

    holo3 = qpretrieve.OffAxisHologram(data.astype(float), copy=True)
    res3 = holo3.run_pipeline(**kwargs)

    assert np.all(res1 == res2)
    assert np.all(res1 == res3)


def test_get_field_sideband(hologram):
    data = hologram
    holo = qpretrieve.OffAxisHologram(data)
    holo.run_pipeline()
    invert_phase = holo.pipeline_kws["invert_phase"]

    kwargs = dict(filter_name="disk",
                  filter_size=1 / 3)

    res1 = holo.run_pipeline(invert_phase=False, **kwargs)
    res2 = holo.run_pipeline(invert_phase=invert_phase, **kwargs)
    assert np.all(res1 == res2)


def test_get_field_three_axes(hologram):
    data1 = hologram
    # create a copy with empty entry in third axis
    data2 = np.zeros((data1.shape[0], data1.shape[1], 2))
    data2[:, :, 0] = data1

    holo1 = qpretrieve.OffAxisHologram(data1)
    holo2 = qpretrieve.OffAxisHologram(data2)

    kwargs = dict(filter_name="disk",
                  filter_size=1 / 3)
    res1 = holo1.run_pipeline(**kwargs)
    res2 = holo2.run_pipeline(**kwargs)
    assert np.all(res1 == res2)


def test_get_field_compare_FFTFilters(hologram):
    data1 = hologram

    holo1 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterNumpy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res1 = holo1.run_pipeline(**kwargs)
    assert res1.shape == (64, 64)

    holo1 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterScipy,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res2 = holo1.run_pipeline(**kwargs)
    assert res2.shape == (64, 64)

    holo1 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterPyFFTW,
                                       padding=False)
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    res3 = holo1.run_pipeline(**kwargs)
    assert res3.shape == (64, 64)

    assert not np.all(res1 == res2)
    assert not np.all(res2 == res3)
