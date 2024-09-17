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
