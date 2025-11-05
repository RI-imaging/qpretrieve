import warnings
from abc import ABC, abstractmethod
from typing import Type

import numpy as np

from ..fourier import get_best_interface, get_available_interfaces
from ..fourier.base import FFTFilter
from ..data_array_layout import (
    convert_data_to_3d_array_layout, convert_3d_data_to_array_layout
)


class BadFFTFilterError(ValueError):
    pass


class BaseInterferogram(ABC):
    default_pipeline_kws = {
        "filter_name": "disk",
        "filter_size": 1 / 3,
        "filter_size_interpretation": "sideband distance",
        "scale_to_filter": False,
        "sideband_freq": None,
        "invert_phase": False,
    }

    def __init__(self, data: np.ndarray,
                 fft_interface: str | Type[FFTFilter] = "auto",
                 subtract_mean=True, padding=2, copy=True,
                 **pipeline_kws) -> None:
        """
        Parameters
        ----------
        data
            The experimental input real-valued image. Allowed input shapes are:
              - 2d (y, x)
              - 3d (z, y, x)
              - rgb (y, x, 3) or rgba (y, x, 4)
        fft_interface
            A Fourier transform interface.
            See :func:`qpretrieve.fourier.get_available_interfaces`
            to get a list of implemented interfaces.
            Default is "auto", which will use
            :func:`qpretrieve.fourier.get_best_interface`. This is in line
            with old behaviour. See Notes for more details.
        subtract_mean: bool
            If True, remove the mean of the hologram before performing
            the Fourier transform. This setting is recommended as it
            can reduce artifacts from frequencies around the central
            band.
        padding: bool
            Boundary padding with zeros; This value determines how
            large the padding region should be. If set to zero, then
            no padding is performed. If set to a positive integer, the
            size is computed to the next power of two (square image)::

                2 ** np.ceil(np.log(padding * max(data.shape) / np.log(2)))
        copy: bool
            Whether to create a copy of the input data.
        pipeline_kws:
            Any additional keyword arguments for :func:`run_pipeline`
            as defined in :const:`default_pipeline_kws`.

        Notes
        -----
        For `fft_interface`, if you do not have the relevant package installed,
        then an error will be raised. For example, setting
        `fft_interface=FFTFilterPyFFTW` will fail if you do not have pyfftw
        installed.

        """
        if fft_interface is None:
            raise BadFFTFilterError(
                "`fft_interface` is set to None or is unavailable."
                "This is likely because you "
                "are trying to use `FFTFilterPyFFTW` or `FFTFilterCupy`. "
                "To use `FFTFilterPyFFTW`, install 'pyfftw'. "
                "To use `FFTFilterCupy`, install 'cupy-cuda12x' or "
                "'cupy-cuda11x', depending on your CUDA version. "
                "If you want qpretrieve to find the best FFT interface "
                "for you, set `fft_interface='auto'`.")
        if fft_interface == 'auto':
            self.ff_iface = get_best_interface()
        else:
            if fft_interface in get_available_interfaces():
                self.ff_iface = fft_interface
            else:
                raise BadFFTFilterError(
                    f"User-chosen FFT Interface '{fft_interface}' is not "
                    f"available. The available interfaces are: "
                    f"{get_available_interfaces()}.\n"
                    f"You can use `fft_interface='auto'` to get the best "
                    f"available interface.")

        # figure out what type of data we have, change it to 3d-stack
        data, self.orig_array_layout = convert_data_to_3d_array_layout(data)

        #: qpretrieve Fourier transform interface class
        self.fft = self.ff_iface(data=data,
                                 subtract_mean=subtract_mean,
                                 padding=padding,
                                 copy=copy)
        #: originally computed Fourier transform
        self.fft_origin = self.fft.fft_origin
        #: filtered Fourier data from last run of `run_pipeline`
        self.fft_filtered = self.fft.fft_filtered
        #: pipeline keyword arguments
        self.pipeline_kws = pipeline_kws
        # Subclasses con override the properties phase, amplitude, and field
        self._field = None
        self._phase = None
        self._amplitude = None

    def get_data_with_input_layout(self, data: np.ndarray | str) -> np.ndarray:
        """Convert `data` to the original input array layout.


        Parameters
        ----------
        data
            Either an array (np.ndarray) or name (str) of the relevant `data`.

        Returns
        -------
        data_out : np.ndarray
            array in the original input array layout

        Notes
        -----
        If `data` is the RGBA array layout, then the alpha (A) channel will be
        an array of ones.

        """
        if isinstance(data, str):
            if data == "fft":
                data = "fft_filtered"
                warnings.warn(
                    "You have asked for 'fft' which is a class. "
                    "Returning 'fft_filtered'. "
                    "Alternatively you could use 'fft_origin'.")
            data = getattr(self, data)
        return convert_3d_data_to_array_layout(data, self.orig_array_layout)

    @property
    def phase(self) -> np.ndarray:
        """Retrieved phase information"""
        if self._phase is None:
            self.run_pipeline()
        return self._phase

    @property
    def amplitude(self) -> np.ndarray:
        """Retrieved amplitude information"""
        if self._amplitude is None:
            self.run_pipeline()
        return self._amplitude

    @property
    def field(self) -> np.ndarray:
        """Retrieved amplitude information"""
        if self._field is None:
            self.run_pipeline()
        return self._field

    def compute_filter_size(
            self,
            filter_size: float,
            filter_size_interpretation: str,
            sideband_freq: tuple[float, float] = None) -> float:
        """Compute the actual filter size in Fourier space"""
        if filter_size_interpretation == "frequency":
            # convert frequency to frequency index
            # We always have padded Fourier data with sizes of order 2.
            fsize = filter_size
        elif filter_size_interpretation == "sideband distance":
            if sideband_freq is None:
                raise ValueError("`sideband_freq` must be set!")
            # filter size based on distance b/w central band and sideband
            if filter_size <= 0 or filter_size >= 1:
                raise ValueError("For sideband distance interpretation, "
                                 "`filter_size` must be between 0 and 1; "
                                 f"got '{filter_size}'!")
            fsize = np.sqrt(np.sum(np.array(sideband_freq) ** 2)) * filter_size
        elif filter_size_interpretation == "frequency index":
            # filter size given in Fourier index (number of Fourier pixels)
            # The user probably does not know that we are padding in
            # Fourier space, so we use the unpadded size and translate it.
            if filter_size <= 0 or filter_size >= self.fft.shape[-2] / 2:
                raise ValueError("For frequency index interpretation, "
                                 + "`filter_size` must be between 0 and "
                                 + f"{self.fft.shape[-2] / 2}, got "
                                 + f"'{filter_size}'!")
            # convert to frequencies (compatible with fx and fy)
            fsize = filter_size / self.fft.shape[-2]
        else:
            raise ValueError("Invalid value for `filter_size_interpretation`: "
                             + f"'{filter_size_interpretation}'")
        return fsize

    def process_like(self, other):
        """Process this dataset in the same way as `other` dataset"""
        assert other.__class__.__name__ == self.__class__.__name__
        self.pipeline_kws.clear()
        if not other.pipeline_kws:
            # run default pipeline
            other.run_pipeline()
        self.run_pipeline(**other.pipeline_kws)

    def get_pipeline_kw(self, key):
        """Current pipeline keyword argument with fallback to defaults"""
        return self.pipeline_kws.get(key, self.default_pipeline_kws[key])

    @abstractmethod
    def run_pipeline(self, **pipeline_kws):
        """Perform pipeline analysis, populating `self.field`

        Parameters
        ----------
        filter_name: str
            specifies the filter to use, see
            :func:`qpretrieve.filter.get_filter_array`.
        filter_size: float
            Size of the filter in Fourier space. The interpretation
            of this value depends on `filter_size_interpretation`.
        filter_size_interpretation: str
            If set to "sideband distance", the filter size is interpreted
            as the relative distance between central band and sideband
            (this is the default). If set to "frequency index", the filter
            size is interpreted as a Fourier frequency index ("pixel size")
            and must be between 0 and `max(hologram.shape)/2`.
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
        sideband_freq: tuple of floats
            Frequency coordinates of the sideband to use. By default,
            a heuristic search for the sideband is done.
        invert_phase: bool
            Invert the phase data.
        """
