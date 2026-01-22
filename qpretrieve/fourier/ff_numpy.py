from .._ndarray_backend import xp
from .base import FFTFilter


class FFTFilterNumpy(FFTFilter):
    """Wraps the numpy Fourier transform
    """
    # always available, because numpy is a dependency
    is_available = True
    expected_backend = "numpy"

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
        return xp.fft.fft2(data, axes=(-2, -1))

    def _ifft(self, data: xp.ndarray) -> xp.ndarray:
        """Perform inverse Fourier transform"""
        return xp.fft.ifft2(data, axes=(-2, -1))
