import numpy as np
import scipy as sp

from qpretrieve import fourier


def test_fft_correct():
    image = np.arange(100).reshape(10, 10)
    ff = fourier.FFTFilterCupy3D(image, subtract_mean=False, padding=False)
    assert np.allclose(
        sp.fft.ifft2(np.fft.ifftshift(ff.fft_origin)).real,
        image,
        rtol=0,
        atol=1e-8
    )


def test_fft_correct_3d():
    image = np.arange(1000).reshape(10, 10, 10)
    ff = fourier.FFTFilterCupy3D(image, subtract_mean=False, padding=False)
    assert np.allclose(
        sp.fft.ifft2(np.fft.ifftshift(ff.fft_origin)).real,
        image,
        rtol=0,
        atol=1e-8
    )


def test_fft_correct_3d_subt_mean():
    subtract_mean = True
    padding = False
    image_3d = np.arange(1000).reshape(10, 10, 10)
    ind = 1
    image_2d = image_3d.copy()[ind]

    ff_cp3d_subt_mean = fourier.FFTFilterCupy3D(
        image_3d, subtract_mean=subtract_mean, padding=padding)
    ff_np2d_subt_mean = fourier.FFTFilterNumpy(
        image_2d, subtract_mean=subtract_mean, padding=padding)

    assert np.allclose(
        ff_cp3d_subt_mean.origin[ind],
        ff_np2d_subt_mean.origin,
        rtol=0,
        atol=1e-8
    )


def test_fft_correct_3d_subt_mean_pad():
    subtract_mean = True
    padding = True
    image_3d = np.arange(1000).reshape(10, 10, 10)
    ind = 1
    image_2d = image_3d.copy()[ind]

    ff_cp3d_subt_mean = fourier.FFTFilterCupy3D(
        image_3d, subtract_mean=subtract_mean, padding=padding)
    ff_np2d_subt_mean = fourier.FFTFilterNumpy(
        image_2d, subtract_mean=subtract_mean, padding=padding)

    assert np.allclose(
        ff_cp3d_subt_mean.origin_padded[ind],
        ff_np2d_subt_mean.origin_padded,
        rtol=0,
        atol=1e-8
    )
