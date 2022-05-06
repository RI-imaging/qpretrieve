from abc import ABC, abstractmethod
import weakref

import numpy as np

from .. import filter


class FFTCache:
    """Cache for Fourier transforms

    This class does not do any Fourier-transforming.

    Whenever a data object is garbage collected, then its
    corresponding Fourier transform is removed from this
    cache.
    """
    cached_output = {}

    @staticmethod
    def add_item(key, data, fft_data):
        weakref.finalize(data, FFTCache.cleanup, key)
        FFTCache.cached_output[key] = fft_data

    @staticmethod
    def get_item(key):
        return FFTCache.cached_output.get(key)

    @staticmethod
    def cleanup(key):
        if key in FFTCache.cached_output:
            FFTCache.cached_output.pop(key)


class FFTFilter(ABC):
    def __init__(self, data, subtract_mean=True, padding=2, copy=True):
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
        padding: int
            Boundary padding with zeros; This value determines how
            large the padding region should be. If set to zero, then
            no padding is performed. If set to a positive integer, the
            size is computed to the next power of two (square image)::

                2 ** np.ceil(np.log(padding * max(data.shape) / np.log(2)))
        copy: bool
            If set to True, make sure that `data` is not edited.
            If you set this to False, then caching FFT results will not
            work anymore.

        Notes
        -----
        The initial Fourier transform of the input data is cached
        using weak references (only if `copy` is True).
        """
        super(FFTFilter, self).__init__()
        # check dtype
        if np.iscomplexobj(data):
            dtype = complex
        else:
            # convert integer-arrays to floating point arrays
            dtype = float
        data_ed = np.array(data, dtype=dtype, copy=copy)
        #: original data (with subtracted mean)
        self.origin = data_ed
        #: whether padding is enabled
        self.padding = padding
        #: whether the mean was subtracted
        self.subtract_mean = subtract_mean
        if subtract_mean:
            # remove contributions of the central band
            # (this affects more than one pixel in the FFT
            # because of zero-padding)
            data_ed -= data_ed.mean()
        if padding:
            # zero padding size is next order of 2
            logfact = np.log(padding * max(data_ed.shape))
            order = int(2 ** np.ceil(logfact / np.log(2)))
            # this is faster than np.pad
            datapad = np.zeros((order, order), dtype=dtype)
            datapad[:data_ed.shape[0], :data_ed.shape[1]] = data_ed
            #: padded input data
            self.origin_padded = datapad
            data_ed = datapad
        else:
            self.origin_padded = None

        # Check if we can used cached data
        weakref_key = "-".join([str(hex(id(data))),
                                str(self.__class__.__name__),
                                str(subtract_mean),
                                str(padding)])
        # Attempt to get the FFT data from a previous run
        fft_data = FFTCache.get_item(weakref_key)
        if fft_data is not None:
            #: frequency-shifted Fourier transform
            self.fft_origin = fft_data
        else:
            #: frequency-shifted Fourier transform
            self.fft_origin = np.fft.fftshift(self._init_fft(data_ed))
            # Add it to the cached FFTs
            if copy:
                FFTCache.add_item(weakref_key, data, self.fft_origin)

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

        Notes
        -----
        The FFT result is cached using weak references in
        :class:`FFTCache`. If you call this function a lot of
        times with different arguments, then it might look like
        a memory leak. However, you just have to delete the
        FFTFilter isntance and everything will get garbage-
        collected.
        """
        weakref_key = "-".join([str(hex(id(self.fft_origin))),
                                str(self.__class__.__name__),
                                str(filter_name),
                                str(filter_size),
                                str(freq_pos),
                                str(self.padding),
                                str(self.shape),
                                str(self.fft_origin.shape)])

        inv_data = FFTCache.get_item(weakref_key)

        if inv_data is not None:
            # Retrieve FFT from cache
            filt_array, field = inv_data
            fft_filtered = self.fft_origin * filt_array
        else:
            filt_array = filter.get_filter_array(
                filter_name=filter_name,
                filter_size=filter_size,
                freq_pos=freq_pos,
                fft_shape=self.fft_origin.shape)
            fft_filtered = self.fft_origin * filt_array
            px = int(freq_pos[0] * self.shape[0])
            py = int(freq_pos[1] * self.shape[1])
            shifted = np.roll(np.roll(fft_filtered, -px, axis=0), -py, axis=1)
            field = self._ifft(np.fft.ifftshift(shifted))
            if self.padding:
                # revert padding
                sx, sy = self.origin.shape
                field = field[:sx, :sy]
            # Add FFT to cache
            # (The cache will only be cleared if this instance is deleted)
            FFTCache.add_item(weakref_key, self.fft_origin,
                              (filt_array, field))

        self.fft_filtered[:] = fft_filtered
        return field
