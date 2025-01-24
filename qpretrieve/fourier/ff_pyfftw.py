import multiprocessing as mp
import numpy as np

import pyfftw

from .base import FFTFilter


class FFTFilterPyFFTW(FFTFilter):
    """Fourier transform using `PyFFTW <https://pyfftw.readthedocs.io/>`_
    """
    is_available = True

    def _init_fft(self, data: np.ndarray) -> np.ndarray:
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
        in_arr = pyfftw.empty_aligned(data.shape, dtype='complex128')
        out_arr = pyfftw.empty_aligned(data.shape, dtype='complex128')
        fft_obj = pyfftw.FFTW(in_arr, out_arr,
                              axes=(-2, -1),
                              threads=mp.cpu_count())
        in_arr[:] = data
        fft_obj()
        return out_arr

    def _ifft(self, data: np.ndarray) -> np.ndarray:
        """Perform inverse Fourier transform"""
        in_arr = pyfftw.empty_aligned(data.shape, dtype='complex128')
        out_arr = pyfftw.empty_aligned(data.shape, dtype='complex128')
        fft_obj = pyfftw.FFTW(in_arr, out_arr, axes=(-2, -1),
                              direction="FFTW_BACKWARD",
                              )
        in_arr[:] = data
        fft_obj()
        return out_arr
