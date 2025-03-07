from __future__ import annotations

from abc import ABC, abstractmethod
import weakref

import numpy as np

from .. import filter
from ..utils import padding_3d, mean_3d
from ..data_array_layout import convert_data_to_3d_array_layout


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
    def __init__(self,
                 data: np.ndarray,
                 subtract_mean: bool = True,
                 padding: int = 2,
                 copy: bool = True) -> None:
        r"""
        Parameters
        ----------
        data
            The experimental input real-valued image. Allowed input shapes are:
              - 2d (y, x)
              - 3d (z, y, x)
              - rgb (y, x, 3) or rgba (y, x, 4)
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
        if not copy:
            # numpy v2.x behaviour requires asarray with copy=False
            copy = None
        data_ed = np.array(data, dtype=dtype, copy=copy)
        # figure out what type of data we have, change it to 3d-stack
        data_ed, self.orig_array_layout = convert_data_to_3d_array_layout(
            data_ed)
        #: original data (with subtracted mean)
        self.origin = data_ed
        # for `subtract_mean` and `padding`, we could use `np.atleast_3d`
        #: whether padding is enabled
        self.padding = padding
        #: whether the mean was subtracted
        self.subtract_mean = subtract_mean
        if subtract_mean:
            # remove contributions of the central band
            # (this affects more than one pixel in the FFT
            # because of zero-padding)
            data_ed = mean_3d(data_ed)
        if padding:
            # zero padding size is next order of 2
            logfact = np.log(padding * max(data_ed.shape))
            order = np.ceil(logfact / np.log(2))
            size = int(2 ** order)

            datapad = padding_3d(data_ed, size, dtype)
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
            self.fft_origin = np.fft.fftshift(
                self._init_fft(data_ed), axes=(-2, -1))
            # Add it to the cached FFTs
            if copy:
                FFTCache.add_item(weakref_key, data, self.fft_origin)

        #: filtered Fourier transform
        self.fft_filtered = np.zeros_like(self.fft_origin)

        #: used Fourier transform (can have a different shape)
        self.fft_used = None

    @property
    def shape(self) -> tuple:
        """Shape of the Fourier transform data"""
        return self.fft_origin.shape

    @property
    @abstractmethod
    def is_available(self) -> bool:
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

    def filter(self, filter_name: str, filter_size: float,
               freq_pos: (float, float),
               scale_to_filter: bool | float = False) -> np.ndarray:
        """
        Parameters
        ----------
        filter_name: str
            specifies the filter to use, one of

            - "disk": binary disk with radius `filter_size`
            - "smooth disk": disk with radius `filter_size` convolved
              with a radial gaussian (`sigma=filter_size/5`)
            - "gauss": radial gaussian (`sigma=0.6*filter_size`)
            - "square": binary square with side length `2*filter_size`
            - "smooth square": square with side length `2*filter_size`
              convolved with square gaussian (`sigma=filter_size/5`)
            - "tukey": a square tukey window of width `2*filter_size` and
              `alpha=0.1`
        filter_size: float
            Size of the filter in Fourier space. The filter size
            interpreted as a Fourier frequency index ("pixel size")
            and must be between 0 and `max(fft_shape)/2`
        freq_pos: tuple of floats
            The position of the filter in frequency coordinates as
            returned by :func:`numpy.fft.fftfreq`.
        scale_to_filter: bool or float
            Crop the image in Fourier space after applying the filter,
            effectively removing surplus (zero-padding) data and
            increasing the pixel size in the output image. If True is
            given, then the cropped area is defined by the filter size,
            if a float is given, the cropped area is defined by the
            filter size multiplied by `scale_to_filter`. You can safely
            set this to True for filters with a binary support. For
            filters such as "smooth square" or "gauss" (filter is not
            a boolean array but a floating-point array), the higher you
            set `scale_to_filter`, the more information will be included
            in the scaled image.

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
                                str(self.fft_origin.shape),
                                str(scale_to_filter),
                                ])

        inv_data = FFTCache.get_item(weakref_key)

        if inv_data is not None:
            # Retrieve FFT from cache
            filt_array, fft_used, field = inv_data
            fft_filtered = self.fft_origin * filt_array
        else:
            filt_array = filter.get_filter_array(
                filter_name=filter_name,
                filter_size=filter_size,
                freq_pos=freq_pos,
                # only take shape of a single fft
                fft_shape=self.fft_origin.shape[-2:])
            fft_filtered = self.fft_origin * filt_array
            px = int(freq_pos[0] * self.shape[-2])
            py = int(freq_pos[1] * self.shape[-1])
            fft_used = np.roll(np.roll(
                fft_filtered, -px, axis=-2), -py, axis=-1)
            if scale_to_filter:
                # Determine the size of the cropping region.
                # We compute the "radius" of the region, so we can
                # crop the data left and right from the center of the
                # Fourier domain.
                osize = fft_filtered.shape[-2]  # square shaped
                crad = int(np.ceil(filter_size * osize * scale_to_filter))
                ccent = osize // 2
                cslice = slice(ccent - crad, ccent + crad)
                # We now have the interesting peak already shifted to
                # the first entry of our array in `shifted`.
                fft_used = fft_used[:, cslice, cslice]

            field = self._ifft(np.fft.ifftshift(fft_used, axes=(-2, -1)))

            if self.padding:
                # revert padding
                sx, sy = self.origin.shape[-2:]
                if scale_to_filter:
                    sx = int(np.ceil(sx * 2 * crad / osize))
                    sy = int(np.ceil(sy * 2 * crad / osize))

                field = field[:, :sx, :sy]

                if scale_to_filter:
                    # Scale the absolute value of the field. This does not
                    # have any influence on the phase, but on the amplitude.
                    field *= (2 * crad / osize) ** 2
            # Add FFT to cache
            # (The cache will only be cleared if this instance is deleted)
            FFTCache.add_item(weakref_key, self.fft_origin,
                              (filt_array, fft_used, field))

        self.fft_filtered[:] = fft_filtered
        self.fft_used = fft_used
        return field
