import scipy as sp
import cupy as cp
import cupyx.scipy.fft as cufft

from .base import FFTFilter


class FFTFilterCupy(FFTFilter):
    """Wraps the cupy Fourier transform and uses it via the scipy backend
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
        data_gpu = cp.asarray(data)
        # likely an inefficiency here, could use `set_global_backend`
        with sp.fft.set_backend(cufft):
            fft_gpu = sp.fft.fft2(data_gpu, axes=(-2, -1))
        fft_cpu = fft_gpu.get()
        return fft_cpu

    def _ifft(self, data):
        """Perform inverse Fourier transform"""
        data_gpu = cp.asarray(data)
        with sp.fft.set_backend(cufft):
            ifft_gpu = sp.fft.ifft2(data_gpu, axes=(-2, -1))
        ifft_cpu = ifft_gpu.get()
        return ifft_cpu
