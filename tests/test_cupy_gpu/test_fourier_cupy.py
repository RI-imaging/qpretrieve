import numpy as np
import scipy as sp

from qpretrieve import fourier

from ..helper_methods import skip_if_missing


@skip_if_missing("cupy")
def test_fft_correct():
    image = np.arange(100).reshape(10, 10)
    ff = fourier.FFTFilterCupy(image, subtract_mean=False, padding=False)
    assert np.allclose(
        sp.fft.ifft2(np.fft.ifftshift(ff.fft_origin)).real,
        image,
        rtol=0,
        atol=1e-8
    )


@skip_if_missing("cupy")
def test_fft_correct_3d():
    image = np.arange(1000).reshape(10, 10, 10)
    ff = fourier.FFTFilterCupy(image, subtract_mean=False, padding=False)

    # use (y, x) axes as in qpretrieve
    axes = (-2, -1)
    ifft_np = np.fft.ifft2(np.fft.ifftshift(
        ff.fft_origin, axes=axes), axes=axes).real

    assert np.allclose(
        ifft_np,
        image,
        rtol=0,
        atol=1e-8
    )
