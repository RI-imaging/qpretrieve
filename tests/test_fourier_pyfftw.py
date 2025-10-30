import numpy as np

from qpretrieve import fourier

from .helper_methods import skip_if_missing


@skip_if_missing("pyfftw")
def test_fft_correct_input_2d():
    image = np.arange(100).reshape(10, 10)
    ff = fourier.FFTFilterPyFFTW(image, subtract_mean=False, padding=False)
    assert ff.fft_origin.shape == (1, 10, 10)
    assert np.allclose(
        np.fft.ifft2(np.fft.ifftshift(ff.fft_origin, axes=(-2, -1))).real,
        image,
        rtol=0,
        atol=1e-8
    )


@skip_if_missing("pyfftw")
def test_fft_correct_input_3d():
    image = np.arange(1000).reshape(10, 10, 10)
    ff = fourier.FFTFilterPyFFTW(image, subtract_mean=False, padding=False)
    assert ff.fft_origin.shape == (10, 10, 10)
    assert np.allclose(
        np.fft.ifft2(np.fft.ifftshift(ff.fft_origin, axes=(-2, -1))).real,
        image,
        rtol=0,
        atol=1e-8
    )


@skip_if_missing("pyfftw")
def test_fft_correct_input_rgb():
    image = np.arange(300).reshape(10, 10, 3)
    ff = fourier.FFTFilterPyFFTW(image, subtract_mean=False, padding=False)
    # does the same as `data_input._convert_rgb_to_3d`
    expected_image = image[:, :, 0][np.newaxis, :, :].copy()
    assert ff.fft_origin.shape == (1, 10, 10)
    assert np.allclose(
        np.fft.ifft2(np.fft.ifftshift(ff.fft_origin, axes=(-2, -1))).real,
        expected_image,
        rtol=0,
        atol=1e-8
    )


@skip_if_missing("pyfftw")
def test_fft_correct_input_rgba():
    image = np.arange(400).reshape(10, 10, 4)
    ff = fourier.FFTFilterPyFFTW(image, subtract_mean=False, padding=False)
    # does the same as `data_input._convert_rgb_to_3d`
    expected_image = image[:, :, 0][np.newaxis, :, :].copy()
    assert ff.fft_origin.shape == (1, 10, 10)
    assert np.allclose(
        np.fft.ifft2(np.fft.ifftshift(ff.fft_origin, axes=(-2, -1))).real,
        expected_image,
        rtol=0,
        atol=1e-8
    )
