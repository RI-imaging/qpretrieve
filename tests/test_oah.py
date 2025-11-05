import numpy as np
import pytest
import pathlib

import qpretrieve
from qpretrieve.interfere import if_oah
from qpretrieve.fourier import FFTFilterNumpy, FFTFilterPyFFTW
from qpretrieve.data_array_layout import (
    convert_data_to_3d_array_layout,
    _convert_2d_to_3d, _convert_3d_to_rgb, _convert_3d_to_rgba,
)

from .helper_methods import skip_if_missing

data_path = pathlib.Path(__file__).parent / "data"


def test_find_sideband():
    size = 40
    ft_data = np.zeros((size, size))
    fx = np.fft.fftshift(np.fft.fftfreq(size))
    ft_data[2, 3] = 1
    ft_data[-3, -2] = 1

    sb1 = if_oah.find_peak_cosine(ft_data=ft_data)
    assert np.allclose(sb1, (fx[2], fx[3]))


def test_fourier2dpad():
    y, x = 100, 120
    data = np.zeros((y, x))
    fft1 = qpretrieve.fourier.FFTFilterNumpy(data)
    assert fft1.shape == (1, 256, 256)

    fft2 = qpretrieve.fourier.FFTFilterNumpy(data, padding=False)
    assert fft2.shape == (1, y, x)


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
    data_2d = hologram
    data_3d, _ = convert_data_to_3d_array_layout(data_2d)
    data_3d = np.repeat(data_3d, repeats=10, axis=0)
    kwargs = dict(sideband=+1,
                  filter_size=1 / 3)
    bad_filter_name = "sepia"

    for data in (data_2d, data_3d):
        holo = qpretrieve.OffAxisHologram(data)

        r_disk = holo.run_pipeline(filter_name="disk", **kwargs)
        r_smdisk = holo.run_pipeline(filter_name="smooth disk", **kwargs)
        r_gauss = holo.run_pipeline(filter_name="gauss", **kwargs)
        r_square = holo.run_pipeline(filter_name="square", **kwargs)
        r_smsquare = holo.run_pipeline(filter_name="smooth square", **kwargs)
        r_tukey = holo.run_pipeline(filter_name="tukey", **kwargs)

        for i in range(len(r_disk)):
            assert np.allclose(
                r_disk[i, 32, 32], 97.307780444912936 - 76.397860381241372j)
            assert np.allclose(
                r_smdisk[i, 32, 32], 108.36438759594623 - 67.1806221692573j)
            assert np.allclose(
                r_gauss[i, 32, 32], 108.2914187451138 - 67.1823527237741j)
            assert np.allclose(
                r_square[i, 32, 32], 102.3285348843612 - 74.139058665601155j)
            assert np.allclose(
                r_smsquare[i, 32, 32], 108.36651862466393 - 67.17988960794392j)
            assert np.allclose(
                r_tukey[i, 32, 32], 113.4826495540899 - 59.546232775481869j)

    with pytest.raises(ValueError, match=f"Unknown filter: {bad_filter_name}"):
        holo.run_pipeline(filter_name=bad_filter_name, **kwargs)


@pytest.mark.parametrize("hologram", [62, 63, 64], indirect=True)
def test_get_field_interpretation_fourier_index(hologram):
    """Filter size in Fourier space using Fourier index new in 0.7.0"""
    data = hologram
    shape_expected = (1, hologram.shape[-2], hologram.shape[-1])
    holo = qpretrieve.OffAxisHologram(data)

    ft_data = holo.fft_origin
    holo.run_pipeline()
    fsx, fsy = holo.pipeline_kws["sideband_freq"]

    kwargs1 = dict(filter_name="disk",
                   filter_size=1 / 3,
                   filter_size_interpretation="sideband distance")
    res1 = holo.run_pipeline(**kwargs1)

    filter_size_fi = np.sqrt(fsx ** 2 + fsy ** 2) / 3 * ft_data.shape[-2]
    kwargs2 = dict(filter_name="disk",
                   filter_size=filter_size_fi,
                   filter_size_interpretation="frequency index",
                   )
    res2 = holo.run_pipeline(**kwargs2)

    assert res1.shape == shape_expected
    assert res2.shape == shape_expected
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

    filter_size_fi = np.sqrt(fsx ** 2 + fsy ** 2) / 3 * ft_data.shape[-2]
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
    assert np.sum(np.sum(mask, axis=-2) != 0) == 17 * 2 + 1


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
    assert np.sum(np.sum(mask, axis=-2) != 0) == 17 * 2 - 1


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
    data2 = np.zeros((data1.shape[0], data1.shape[1], 3))
    data2[:, :, 0] = data1
    # both will be output as (z,y,x) shaped image stacks
    shape_expected = (1, hologram.shape[-2], hologram.shape[-1])

    holo1 = qpretrieve.OffAxisHologram(data1)
    holo2 = qpretrieve.OffAxisHologram(data2)

    kwargs = dict(filter_name="disk",
                  filter_size=1 / 3)
    res1 = holo1.run_pipeline(**kwargs)
    res2 = holo2.run_pipeline(**kwargs)

    assert res1.shape == shape_expected
    assert res2.shape == shape_expected
    assert np.all(res1 == res2)


@skip_if_missing("pyfftw")
def test_get_field_compare_FFTFilters(hologram):
    data1 = hologram
    kwargs = dict(filter_name="disk", filter_size=1 / 3)
    padding = False
    shape_expected = (1, hologram.shape[-2], hologram.shape[-1])

    holo1 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterNumpy,
                                       padding=padding)
    res1 = holo1.run_pipeline(**kwargs)
    assert res1.shape == shape_expected

    holo2 = qpretrieve.OffAxisHologram(data1,
                                       fft_interface=FFTFilterPyFFTW,
                                       padding=padding)
    res2 = holo2.run_pipeline(**kwargs)
    assert res2.shape == shape_expected

    # not exactly the same, but roughly equal to 1e-5
    assert np.allclose(holo1.fft.fft_used, holo2.fft.fft_used)
    assert np.allclose(res1, res2)


def test_field_format_consistency(hologram):
    """The data format returned should be (z,y,x)"""
    data_2d = hologram.copy()
    shape_expected = (1, hologram.shape[-2], hologram.shape[-1])

    # 2d data format
    holo_2d = qpretrieve.OffAxisHologram(data_2d)
    res_2d = holo_2d.run_pipeline()
    assert res_2d.shape == shape_expected

    # 3d data format
    data_3d, _ = _convert_2d_to_3d(data_2d)
    holo_3d = qpretrieve.OffAxisHologram(data_3d)
    res_3d = holo_3d.run_pipeline()
    assert res_3d.shape == shape_expected

    # rgb data format
    data_rgb = _convert_3d_to_rgb(data_3d)
    holo_rgb = qpretrieve.OffAxisHologram(data_rgb)
    res_rgb = holo_rgb.run_pipeline()
    assert res_rgb.shape == shape_expected

    # rgba data format
    data_rgba = _convert_3d_to_rgba(data_3d)
    holo_rgba = qpretrieve.OffAxisHologram(data_rgba)
    res_rgba = holo_rgba.run_pipeline()
    assert res_rgba.shape == shape_expected

    assert np.all(res_2d == res_3d)
    assert np.all(res_2d == res_rgb)
    assert np.all(res_2d == res_rgba)


def test_oah_2d_vs_3d_processing():
    edata_2d = np.load(data_path / "hologram_cell.npz")["data"]
    edata_3d, _ = convert_data_to_3d_array_layout(edata_2d)
    edata_3d = np.repeat(edata_3d, repeats=5, axis=0)

    # 3d
    holo_3d = qpretrieve.OffAxisHologram(data=edata_3d, padding=True)
    fields_3d = holo_3d.run_pipeline()

    # 2d
    fields_2d = np.zeros_like(fields_3d)
    for i in range(fields_2d.shape[0]):
        holo = qpretrieve.OffAxisHologram(data=edata_2d, padding=True)
        holo.run_pipeline()
        fields_2d[i] = holo.field[0]  # there is only 1 per input

    assert fields_2d.shape == fields_3d.shape
    assert np.array_equal(fields_2d, fields_3d)
