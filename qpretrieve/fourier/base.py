from abc import ABC, abstractmethod

import numpy as np

from .. import filter


class FFTFilter(ABC):
    def __init__(self, data, subtract_mean=True, padding=True, copy=True):
        r"""
        Parameters
        ----------
        data: 2d real-valued np.ndarray
            The experimental input image
        subtract_mean: bool
            If True, subtract the mean of `data` before performing
            the Fourier transform. This setting is recommended as it
            can reduce artifacts from frequencies around the central
            band.
        padding: bool
            Whether to perform boundary-padding with linear ramp
        copy: bool
            If set to True, make sur that `data` is not edited.
        """
        super(FFTFilter, self).__init__()
        if np.iscomplexobj(data):
            dtype = complex
        else:
            # convert integer-arrays to floating point arrays
            dtype = float
        data = np.array(data, dtype=dtype, copy=copy)
        #: original data (with subtracted mean)
        self.origin = data
        #: whether padding is enabled
        self.padding = padding
        if subtract_mean:
            # remove contributions of the central band
            # (this affects more than one pixel in the FFT
            # because of zero-padding)
            data -= data.mean()
        if padding:
            # zero padding size is next order of 2
            (N, M) = data.shape
            order = int(
                max(64., 2 ** np.ceil(np.log(2 * max(N, M)) / np.log(2))))

            # this is faster than np.pad
            datapad = np.zeros((order, order), dtype=float)
            datapad[:data.shape[0], :data.shape[1]] = data
            #: padded input data
            self.origin_padded = datapad
            data = datapad
        else:
            self.origin_padded = None
        #: frequency-shifted Fourier transform
        self.fft_origin = np.fft.fftshift(self._init_fft(data))
        #: filtered Fourier transform
        self.fft_filtered = np.zeros_like(self.fft_origin)

    @property
    def shape(self):
        """Shape of the Fourier transform data"""
        return self.fft_origin.shape

    @property
    @abstractmethod
    def is_available(self):
        """Whether this method is available given current hardware/software"""
        return True

    @abstractmethod
    def _ifft(self, data):
        """Perform inverse Fourier transform"""

    @abstractmethod
    def _init_fft(self, data):
        """Initialize Fourier transform

        This is where you would compute the initial Fourier transform.
        E.g. for FFTW, you would do planning here.

        Parameters
        ----------
        data: 2d real-valued np.ndarray
            Input field to be refocused

        Returns
        -------
        fft_fdata: 2d complex-valued ndarray
            Fourier transform `data`
        """

    def filter(self, filter_name, filter_size, freq_pos):
        """

        Parameters
        ----------
        filter_name: str
            specifies the filter to use, one of

            - "disk": binary disk with radius `filter_size`
            - "smooth disk": disk with radius `filter_size` convolved
              with a radial gaussian (`sigma=filter_size/5`)
            - "gauss": radial gaussian (`sigma=0.6*filter_size`)
            - "square": binary square with side length `filter_size`
            - "smooth square": square with side length `filter_size`
              convolved with square gaussian (`sigma=filter_size/5`)
            - "tukey": a square tukey window of width `2*filter_size` and
              `alpha=0.1`
        filter_size: float
            Size of the filter in Fourier space. The filter size
            interpreted as a Fourier frequency index ("pixel size")
            and must be between 0 and `max(fft_shape)/2`
        freq_pos: tuple of floats
            The position of the filter in frequency coordinates as
            returned by :func:`nunpy.fft.fftfreq`.
        """
        filt_array = filter.get_filter_array(
            filter_name=filter_name,
            filter_size=filter_size,
            freq_pos=freq_pos,
            fft_shape=self.fft_origin.shape)

        self.fft_filtered[:] = self.fft_origin * filt_array
        px = int(freq_pos[0] * self.shape[0])
        py = int(freq_pos[1] * self.shape[1])
        shifted = np.roll(np.roll(self.fft_filtered, -px, axis=0), -py, axis=1)
        field = self._ifft(np.fft.ifftshift(shifted))
        if self.padding:
            sx, sy = self.origin.shape
            field = field[:sx, :sy]
        return field
