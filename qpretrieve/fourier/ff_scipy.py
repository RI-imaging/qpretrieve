import scipy as sp


from .base import FFTFilter


class FFTFilterScipy(FFTFilter):
    """Wraps the scipy Fourier transform
    """
    # always available, because scipy is a dependency
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
        return sp.fft.fft2(data)

    def _ifft(self, data):
        """Perform inverse Fourier transform"""
        return sp.fft.ifft2(data)
