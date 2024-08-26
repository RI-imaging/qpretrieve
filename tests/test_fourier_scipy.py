import numpy as np
import scipy as sp

from qpretrieve import fourier


def test_fft_correct():
    image = np.arange(100).reshape(10, 10)
    ff = fourier.FFTFilterScipy(image, subtract_mean=False, padding=False)
    assert np.allclose(
        sp.fft.ifft2(np.fft.ifftshift(ff.fft_origin)).real,
        image,
        rtol=0,
        atol=1e-8
    )
