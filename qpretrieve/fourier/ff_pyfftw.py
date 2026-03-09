import multiprocessing as mp
import pyfftw

from .._ndarray_backend import xp
from .base import FFTFilter


class FFTFilterPyFFTW(FFTFilter):
    """Fourier transform using `PyFFTW <https://pyfftw.readthedocs.io/>`_
    """
    is_available = True

    def _init_fft(self, data: xp.ndarray) -> xp.ndarray:
        """Perform initial Fourier transform of the input data

        Parameters
        ----------
        data: 2d real-valued xp.ndarray
            Input field to be refocused

        Returns
        -------
        fft_fdata: 2d complex-valued ndarray
            Fourier transform `data`
        """
        dtype_out = self._result_type(data.dtype)
        in_arr = pyfftw.empty_aligned(data.shape, dtype=dtype_out)
        out_arr = pyfftw.empty_aligned(data.shape, dtype=dtype_out)
        fft_obj = pyfftw.FFTW(in_arr, out_arr,
                              axes=(-2, -1),
                              threads=mp.cpu_count())
        in_arr[:] = data
        fft_obj()
        return out_arr

    def _ifft(self, data: xp.ndarray) -> xp.ndarray:
        """Perform inverse Fourier transform"""
        dtype_out = self._result_type(data.dtype)
        in_arr = pyfftw.empty_aligned(data.shape, dtype=dtype_out)
        out_arr = pyfftw.empty_aligned(data.shape, dtype=dtype_out)
        fft_obj = pyfftw.FFTW(in_arr, out_arr, axes=(-2, -1),
                              direction="FFTW_BACKWARD",
                              )
        in_arr[:] = data
        fft_obj()
        return out_arr

    def _result_type(self, dtype_in) -> xp.dtype:
        """Wrapper on `np.result_type` to provide correct fft dtype"""
        dtype_out = xp.complex128
        if dtype_in != xp.float64:
            dtype_out = xp.result_type(
                xp.fft.fft(xp.arange(1, dtype=dtype_in)))
        return dtype_out
