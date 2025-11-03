import pytest
import copy
import pathlib

import h5py
import numpy as np
from skimage.restoration import unwrap_phase
from scipy import signal

from qpretrieve import fourier, interfere

from .helper_methods import skip_if_missing

data_path = pathlib.Path(__file__).parent / "data"


def test_scale_sanity_check():
    """Scale an image and compare with skimage"""
    # create a 2D gaussian test image
    x = np.linspace(-100, 100, 100)
    xx, yy = np.meshgrid(x, -x, indexing="ij")
    gauss = np.exp(-(xx ** 2 + yy ** 2) / 625)

    ft = fourier.FFTFilterNumpy(gauss, subtract_mean=False)

    # That's basically a sanity check. The IFT(FT(gauss)) == gauss
    if1 = ft.filter(filter_name="disk",
                    filter_size=1,
                    freq_pos=(0, 0),
                    scale_to_filter=False)
    assert np.allclose(gauss, if1, rtol=0, atol=1e-15)

    # Now we reduce the filter radius to 0.25. This should still yield
    # good results, because the gaussian is small. If you look at the
    # difference image, you will see rings now.
    if2 = ft.filter(filter_name="disk",
                    filter_size=0.25,
                    freq_pos=(0, 0),
                    scale_to_filter=False)
    assert np.allclose(gauss, if2, rtol=0, atol=4e-4)

    # Now scale to the filter size:
    if3 = ft.filter(filter_name="disk",
                    filter_size=0.25,
                    freq_pos=(0, 0),
                    scale_to_filter=True)
    # (and scale the input image as well)
    gauss3 = signal.resample(signal.resample(gauss, 50, axis=0),
                             50, axis=1)
    assert np.allclose(gauss3, if3, rtol=0, atol=1e-7)


def test_scale_sanity_check_other_filter():
    """Scale an image and compare with skimage"""
    # create a 2D gaussian test image
    x = np.linspace(-100, 100, 100)
    xx, yy = np.meshgrid(x, -x, indexing="ij")
    gauss = np.exp(-(xx ** 2 + yy ** 2) / 625)

    ft = fourier.FFTFilterNumpy(gauss, subtract_mean=False)

    # That's basically a sanity check. The IFT(FT(gauss)) == gauss
    if1 = ft.filter(filter_name="disk",
                    filter_size=1,
                    freq_pos=(0, 0),
                    scale_to_filter=False)
    assert np.allclose(gauss, if1, rtol=0, atol=1e-15)

    # Now we reduce the filter radius to 0.25. This should still yield
    # good results, because the gaussian is small. If you look at the
    # difference image, you will see rings now.
    if2 = ft.filter(filter_name="disk",
                    filter_size=0.125,
                    freq_pos=(0, 0),
                    scale_to_filter=False)
    assert np.allclose(gauss, if2, rtol=0, atol=4e-4)

    # Now scale to the filter size:
    if3 = ft.filter(filter_name="disk",
                    filter_size=0.125,
                    freq_pos=(0, 0),
                    scale_to_filter=True)
    # (and scale the input image as well)
    gauss3 = signal.resample(
        signal.resample(gauss, 25, axis=0),
        25, axis=1)
    assert np.allclose(gauss3, if3, rtol=0, atol=1e-7)


def test_scale_to_filter_oah():
    data = np.load(data_path / "hologram_cell.npz")
    image = data["data"]
    refer = data["bg_data"]

    # Standard analysis pipeline
    pipeline_kws = {
        'filter_name': 'disk',
        'filter_size': 0.3333333333333333,
        'filter_size_interpretation': 'sideband distance',
        'scale_to_filter': False,
        'sideband_freq': (-0.203125, -0.12109375),
        'invert_phase': False
    }

    ifh = interfere.OffAxisHologram(image, **pipeline_kws)
    ifh.run_pipeline()

    ifr = interfere.OffAxisHologram(refer, **pipeline_kws)
    ifr.run_pipeline()

    phase = unwrap_phase(ifh.phase - ifr.phase)
    assert phase.shape == (1, 200, 210)
    assert np.allclose(phase.mean(), 1.0840394954441188, atol=1e-5)

    # Rescaled pipeline
    pipeline_kws_scale = copy.copy(pipeline_kws)
    pipeline_kws_scale["scale_to_filter"] = True
    ifh.run_pipeline(**pipeline_kws_scale)
    ifr.run_pipeline(**pipeline_kws_scale)
    phase_scaled = unwrap_phase(ifh.phase - ifr.phase)
    assert phase_scaled.shape == (1, 33, 34)
    assert np.allclose(phase_scaled.mean(), 1.0469570087033453, atol=1e-5)


def test_bad_fft_interface_input():
    """Fails because inputting PyFFTW without installing defaults to None"""
    data = np.load(data_path / "hologram_cell.npz")
    image = data["data"]

    with pytest.raises(
            interfere.BadFFTFilterError,
            match="`fft_interface` is set to None or is unavailable."
                  "This is likely because you "
                  "are trying to use `FFTFilterPyFFTW` or `FFTFilterCupy`. "
                  "To use `FFTFilterPyFFTW`, install 'pyfftw'. "
                  "To use `FFTFilterCupy`, install 'cupy-cuda12x' or "
                  "'cupy-cuda11x', depending on your CUDA version. "
                  "If you want qpretrieve to find the best FFT interface "
                  "for you, set `fft_interface='auto'`."):
        interfere.OffAxisHologram(image, fft_interface=None)


def test_scale_to_filter_qlsi():
    with h5py.File(data_path / "qlsi_paa_bead.h5") as h5:
        image = h5["0"][:]
        refer = h5["reference"][:]

    # Standard analysis pipeline
    pipeline_kws = {
        'wavelength': 550e-9,
        'qlsi_pitch_term': 1.87711e-08,
        'filter_name': "disk",
        'filter_size': 180,
        'filter_size_interpretation': "frequency index",
        'scale_to_filter': False,
        'invert_phase': False
    }

    ifh = interfere.QLSInterferogram(image, **pipeline_kws)
    raw_wavefront = ifh.run_pipeline()
    assert raw_wavefront.shape == (1, 720, 720)
    assert ifh.phase.shape == (1, 720, 720)
    assert ifh.amplitude.shape == (1, 720, 720)
    assert ifh.field.shape == (1, 720, 720)

    ifr = interfere.QLSInterferogram(refer, **pipeline_kws)
    ifr.run_pipeline()
    assert ifr.phase.shape == (1, 720, 720)
    assert ifr.amplitude.shape == (1, 720, 720)
    assert ifr.field.shape == (1, 720, 720)

    ifh_phase = ifh.phase[0]
    ifr_phase = ifr.phase[0]

    phase = unwrap_phase(ifh_phase - ifr_phase)

    assert phase.shape == (720, 720)
    assert np.allclose(phase.mean(), 0.12434563269684816, atol=1e-6)

    # Rescaled pipeline
    pipeline_kws_scale = copy.copy(pipeline_kws)
    pipeline_kws_scale["scale_to_filter"] = True
    ifh.run_pipeline(**pipeline_kws_scale)
    ifr.run_pipeline(**pipeline_kws_scale)
    phase_scaled = unwrap_phase(ifh.phase - ifr.phase)

    assert phase_scaled.shape == (1, 126, 126)

    assert np.allclose(phase_scaled.mean(), 0.1257080793074251, atol=1e-6)


def test_fft_dimensionality_consistency():
    """Compare using fft algorithms on 2d and 3d data."""
    image_3d = np.arange(1000).reshape(10, 10, 10)
    image_2d = image_3d[0].copy()

    # fft with shift
    fft_3d = np.fft.fftshift(np.fft.fft2(image_3d, axes=(-2, -1)),
                             axes=(-2, -1))
    fft_2d = np.fft.fftshift(np.fft.fft2(image_2d))  # old qpretrieve
    assert fft_3d.shape == (10, 10, 10)
    assert fft_2d.shape == (10, 10)
    assert np.allclose(fft_3d[0], fft_2d, rtol=0, atol=1e-8)

    # ifftshift
    fft_3d_shifted = np.fft.ifftshift(fft_3d, axes=(-2, -1))
    fft_2d_shifted = np.fft.ifftshift(fft_2d)  # old qpretrieve
    assert fft_3d_shifted.shape == (10, 10, 10)
    assert fft_2d_shifted.shape == (10, 10)
    assert np.allclose(fft_3d_shifted[0], fft_2d_shifted, rtol=0, atol=1e-8)

    # ifft
    ifft_3d_shifted = np.fft.ifft2(fft_3d_shifted, axes=(-2, -1))
    ifft_2d_shifted = np.fft.ifft2(fft_2d_shifted)  # old qpretrieve
    assert ifft_3d_shifted.shape == (10, 10, 10)
    assert ifft_2d_shifted.shape == (10, 10)
    assert np.allclose(ifft_3d_shifted[0], ifft_2d_shifted, rtol=0, atol=1e-8)

    assert np.allclose(ifft_3d_shifted.real, image_3d, rtol=0, atol=1e-8)
    assert np.allclose(ifft_2d_shifted.real, image_2d, rtol=0, atol=1e-8)


@skip_if_missing("pyfftw")
def test_fft_comparison_FFTFilter():
    image = np.arange(1000).reshape(10, 10, 10)
    ff_np = fourier.FFTFilterNumpy(image, subtract_mean=False, padding=False)
    ff_tw = fourier.FFTFilterPyFFTW(image, subtract_mean=False, padding=False)
    assert ff_np.fft_origin.shape == ff_tw.fft_origin.shape == (10, 10, 10)

    assert np.allclose(ff_np.fft_origin, ff_tw.fft_origin, rtol=0, atol=1e-8)
    assert np.allclose(
        np.fft.ifft2(np.fft.ifftshift(ff_np.fft_origin, axes=(-2, -1))).real,
        np.fft.ifft2(np.fft.ifftshift(ff_tw.fft_origin, axes=(-2, -1))).real,
        rtol=0,
        atol=1e-8
    )


@skip_if_missing("pyfftw")
def test_fft_comparison_data_input_fmt():
    image = np.arange(1000).reshape(10, 10, 10)
    FFTFilters = [fourier.FFTFilterNumpy, fourier.FFTFilterPyFFTW]

    for fftfilt in FFTFilters:
        # 3d input
        ff_3d = fftfilt(image, subtract_mean=False, padding=False)
        # 2d input
        ff_arr_2d = np.zeros_like(ff_3d.fft_origin)
        for i, img in enumerate(image):
            ff_2d = fftfilt(img, subtract_mean=False, padding=False)
            ff_arr_2d[i] = ff_2d.fft_origin

            # ffts are the same
            assert np.allclose(ff_2d.fft_origin,
                               ff_3d.fft_origin[i],
                               rtol=0, atol=1e-8)
            # iffts are the same
            assert np.allclose(np.fft.ifft2(ff_2d.fft_origin).real,
                               np.fft.ifft2(ff_3d.fft_origin[i]).real,
                               rtol=0, atol=1e-8)
            # shifted iffts are the same, if you use arg axes
            assert np.allclose(
                np.fft.ifft2(np.fft.ifftshift(
                    ff_2d.fft_origin, axes=(-2, -1))).real,
                np.fft.ifft2(np.fft.ifftshift(
                    ff_3d.fft_origin[i], axes=(-2, -1))).real,
                rtol=0, atol=1e-8)
            # shifted 2d ifft is the same as the 2d img
            assert np.allclose(
                np.fft.ifft2(np.fft.ifftshift(
                    ff_2d.fft_origin, axes=(-2, -1))).real,
                img, rtol=0, atol=1e-8)

        # shifted 3d ifft is the same as the 3d img
        assert np.allclose(
            np.fft.ifft2(np.fft.ifftshift(ff_3d.fft_origin,
                                          axes=(-2, -1))).real,
            image, rtol=0, atol=1e-8)
