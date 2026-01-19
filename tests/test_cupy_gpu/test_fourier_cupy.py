import numpy as np
import scipy as sp

from qpretrieve import fourier
from qpretrieve._ndarray_backend import xp

from ..helper_methods import skip_if_missing


@skip_if_missing("cupy")
def test_fft_correct(set_ndarray_backend_to_cupy):
    image = np.arange(100).reshape(10, 10)
    ff = fourier.FFTFilterCupy(image, subtract_mean=False, padding=False)

    # assert on gpu
    image_gpu = xp.asarray(image)
    assert np.allclose(
        xp.fft.ifft2(xp.fft.ifftshift(ff.fft_origin)).real,
        image_gpu, rtol=0, atol=1e-13)

    # assert on cpu
    fft_origin = ff.fft_origin.get()
    assert np.allclose(
        sp.fft.ifft2(np.fft.ifftshift(fft_origin)).real,
        image, rtol=0, atol=1e-13)


@skip_if_missing("cupy")
def test_fft_correct_3d(set_ndarray_backend_to_cupy):
    image = np.arange(1000).reshape(10, 10, 10)
    ff = fourier.FFTFilterCupy(image, subtract_mean=False, padding=False)

    # assert on cpu
    fft_origin = ff.fft_origin.get()

    # use (y, x) axes as in qpretrieve
    axes = (-2, -1)
    ifft_np = np.fft.ifft2(np.fft.ifftshift(
        fft_origin, axes=axes), axes=axes).real

    assert np.allclose(ifft_np, image, rtol=0, atol=1e-12)
