import numpy as np


from .base import FFTFilter


class FFTFilterNumpy(FFTFilter):
    """Wraps the numpy Fourier transform
    """
    # always available, because numpy is a dependency
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
        return np.fft.fft2(data, axes=(-2, -1))

    def _ifft(self, data: np.ndarray) -> np.ndarray:
        """Perform inverse Fourier transform"""
        return np.fft.ifft2(data, axes=(-2, -1))
