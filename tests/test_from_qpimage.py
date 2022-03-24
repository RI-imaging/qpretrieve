"""These are tests from qpimage"""
import numpy as np
import pytest

import qpretrieve
from qpretrieve.interfere import if_oadhm


def hologram(size=64):
    x = np.arange(size).reshape(-1, 1) - size / 2
    y = np.arange(size).reshape(1, -1) - size / 2

    amp = np.linspace(.9, 1.1, size * size).reshape(size, size)
    pha = np.linspace(0, 2, size * size).reshape(size, size)

    rad = x**2 + y**2 > (size / 3)**2
    pha[rad] = 0
    amp[rad] = 1

    # frequencies must match pixel in Fourier space
    kx = 2 * np.pi * -.3
    ky = 2 * np.pi * -.3
    image = (amp**2 + np.sin(kx * x + ky * y + pha) + 1) * 255
    return image


def test_find_sideband():
    size = 40
    ft_data = np.zeros((size, size))
    fx = np.fft.fftshift(np.fft.fftfreq(size))
    ft_data[2, 3] = 1
    ft_data[-3, -2] = 1

    sb1 = if_oadhm.find_peak_cosine(ft_data=ft_data)
    assert np.allclose(sb1, (fx[2], fx[3]))


def test_fourier2dpad():
    data = np.zeros((100, 120))
    fft1 = qpretrieve.fourier.FFTFilterNumpy(data)
    assert fft1.shape == (256, 256)

    fft2 = qpretrieve.fourier.FFTFilterNumpy(data, padding=False)
    assert fft2.shape == data.shape


def test_get_field_error_bad_filter_size():
    data = hologram()

    holo = qpretrieve.OffAxisHologram(data)
    with pytest.raises(ValueError, match="must be between 0 and 1"):
        holo.run_pipeline(filter_size=2)


def test_get_field_error_bad_filter_size_interpretation_frequency_index():
    data = hologram(size=64)
    holo = qpretrieve.OffAxisHologram(data)

    with pytest.raises(ValueError,
                       match=r"must be between 0 and 64"):
        holo.run_pipeline(filter_size_interpretation="frequency index",
                          filter_size=64)


def test_get_field_error_invalid_interpretation():
    data = hologram()
    holo = qpretrieve.OffAxisHologram(data)

    with pytest.raises(ValueError,
                       match="Invalid value for `filter_size_interpretation`"):
        holo.run_pipeline(filter_size_interpretation="blequency")


def test_get_field_filter_names():
    data = hologram()
    holo = qpretrieve.OffAxisHologram(data)

    kwargs = dict(sideband=+1,
                  filter_size=1 / 3)

    r_disk = holo.run_pipeline(filter_name="disk", **kwargs)
    assert np.allclose(
        r_disk[32, 32], 97.307780444912936 - 76.397860381241372j)

    r_smooth_disk = holo.run_pipeline(filter_name="smooth disk", **kwargs)
    assert np.allclose(r_smooth_disk[32, 32],
                       108.36438759594623-67.1806221692573j)

    r_gauss = holo.run_pipeline(filter_name="gauss", **kwargs)
    assert np.allclose(r_gauss[32, 32],
                       108.2914187451138-67.1823527237741j)

    r_square = holo.run_pipeline(filter_name="square", **kwargs)
    assert np.allclose(
        r_square[32, 32], 102.3285348843612 - 74.139058665601155j)

    r_smsquare = holo.run_pipeline(filter_name="smooth square", **kwargs)
    assert np.allclose(
        r_smsquare[32, 32], 105.23157221309754 - 70.593282942004862j)

    r_tukey = holo.run_pipeline(filter_name="tukey", **kwargs)
    assert np.allclose(
        r_tukey[32, 32], 113.4826495540899 - 59.546232775481869j)

    try:
        holo.run_pipeline(filter_name="unknown", **kwargs)
    except ValueError:
        pass
    else:
        assert False, "unknown filter accepted"


@pytest.mark.parametrize("size", [62, 63, 64])
def test_get_field_interpretation_fourier_index(size):
    """Filter size in Fourier space using Fourier index new in 0.7.0"""
    data = hologram(size=size)
    holo = qpretrieve.OffAxisHologram(data)

    ft_data = holo.fft_origin
    holo.run_pipeline()
    fsx, fsy = holo.pipeline_kws["sideband_freq"]

    kwargs1 = dict(filter_name="disk",
                   filter_size=1/3,
                   filter_size_interpretation="sideband distance")
    res1 = holo.run_pipeline(**kwargs1)

    filter_size_fi = np.sqrt(fsx**2 + fsy**2) / 3 * ft_data.shape[0]
    kwargs2 = dict(filter_name="disk",
                   filter_size=filter_size_fi,
                   filter_size_interpretation="frequency index",
                   )
    res2 = holo.run_pipeline(**kwargs2)
    assert np.all(res1 == res2)


@pytest.mark.parametrize("size", [62, 63, 64])
def test_get_field_interpretation_fourier_index_control(size):
    """Filter size in Fourier space using Fourier index new in 0.7.0"""
    data = hologram(size=size)
    holo = qpretrieve.OffAxisHologram(data)

    ft_data = holo.fft_origin
    holo.run_pipeline()
    fsx, fsy = holo.pipeline_kws["sideband_freq"]

    evil_factor = 1.1

    kwargs1 = dict(filter_name="disk",
                   filter_size=1/3 * evil_factor,
                   filter_size_interpretation="sideband distance"
                   )
    res1 = holo.run_pipeline(**kwargs1)

    filter_size_fi = np.sqrt(fsx**2 + fsy**2) / 3 * ft_data.shape[0]
    kwargs2 = dict(filter_name="disk",
                   filter_size=filter_size_fi,
                   filter_size_interpretation="frequency index",
                   )
    res2 = holo.run_pipeline(**kwargs2)
    assert not np.all(res1 == res2)


@pytest.mark.parametrize("size", [62, 63, 64, 134, 135])
@pytest.mark.parametrize("filter_size", [17, 17.01])
def test_get_field_interpretation_fourier_index_mask_1(size, filter_size):
    """Make sure filter size in Fourier space pixels is correct"""
    data = hologram(size=size)
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
    assert np.sum(np.sum(mask, axis=0) != 0) == 17*2 + 1


@pytest.mark.parametrize("size", [62, 63, 64, 134, 135])
def test_get_field_interpretation_fourier_index_mask_2(size):
    """Filter size in Fourier space using Fourier index new in 0.7.0"""
    data = hologram(size=size)
    holo = qpretrieve.OffAxisHologram(data)

    kwargs2 = dict(filter_name="disk",
                   filter_size=16.99,
                   filter_size_interpretation="frequency index"
                   )
    holo.run_pipeline(**kwargs2)
    mask = holo.fft_filtered.real != 0

    # We get two points less than in the previous test, because we
    # loose on each side of the spectrum.
    assert np.sum(np.sum(mask, axis=0) != 0) == 17*2 - 1


def test_get_field_int_copy():
    data = hologram()
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


def test_get_field_sideband():
    data = hologram()
    holo = qpretrieve.OffAxisHologram(data)
    holo.run_pipeline()
    invert_phase = holo.pipeline_kws["invert_phase"]

    kwargs = dict(filter_name="disk",
                  filter_size=1 / 3)

    res1 = holo.run_pipeline(invert_phase=False, **kwargs)
    res2 = holo.run_pipeline(invert_phase=invert_phase, **kwargs)
    assert np.all(res1 == res2)


def test_get_field_three_axes():
    data1 = hologram()
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
