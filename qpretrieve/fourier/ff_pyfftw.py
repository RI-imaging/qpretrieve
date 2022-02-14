import pyfftw

from .base import FFTFilter


class FFTFilterPyFFTW(FFTFilter):
    """Wrapper for numpy Fourier transform
    """
    # always available, because numpy is a dependency
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
        in_arr = pyfftw.empty_aligned(data.shape, dtype='complex128')
        out_arr = pyfftw.empty_aligned(data.shape, dtype='complex128')
        fft_obj = pyfftw.FFTW(in_arr, out_arr, axes=(0, 1))
        in_arr[:] = data
        fft_obj()
        return out_arr

    def _ifft(self, data):
        """Perform inverse Fourier transform"""
        in_arr = pyfftw.empty_aligned(data.shape, dtype='complex128')
        ou_arr = pyfftw.empty_aligned(data.shape, dtype='complex128')
        fft_obj = pyfftw.FFTW(in_arr, ou_arr, axes=(0, 1),
                              direction="FFTW_BACKWARD",
                              )
        in_arr[:] = data
        fft_obj()
        return ou_arr
