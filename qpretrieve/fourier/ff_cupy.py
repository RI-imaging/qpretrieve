import warnings
import scipy as sp
from .. import _ndarray_backend as xp
import cupyx.scipy.fft as cufft

from .base import FFTFilter
from .._ndarray_backend import NDArrayBackendWarning


class FFTFilterCupy(FFTFilter):
    """Wraps the cupy Fourier transform and uses it via the scipy backend

    .. versionadded:: 0.5.0

    """
    is_available = True

    def _init_fft(self, data):
        """Perform initial Fourier transform of the input data

        Parameters
        ----------
        data: 2d real-valued np.ndarray
            Input field to be refocused

        Returns
        -------
        fft_fdata: 2d complex-valued ndarray
            Fourier transform `data`
        """
        if not xp._is_cupy():
            warnings.warn(NDArrayBackendWarning(
                "You are using `FFTFilterCupy` without the 'cupy' ndarray "
                "backend. This will limit the FFT speed. To set the ndarray "
                "backend, use `qpretrieve.set_ndarray_backend('cupy')` "))
        data_gpu = xp.asarray(data)
        # likely an inefficiency here, could use `set_global_backend`
        with sp.fft.set_backend(cufft):
            fft_gpu = sp.fft.fft2(data_gpu, axes=(-2, -1))
        # fft_cpu = fft_gpu.get()
        # return fft_cpu
        return fft_gpu

    def _ifft(self, data):
        """Perform inverse Fourier transform"""
        data_gpu = xp.asarray(data)
        with sp.fft.set_backend(cufft):
            ifft_gpu = sp.fft.ifft2(data_gpu, axes=(-2, -1))
        # ifft_cpu = ifft_gpu.get()
        # return ifft_cpu
        return ifft_gpu
