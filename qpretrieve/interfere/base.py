from abc import ABC, abstractmethod

import numpy as np

from ..fourier import get_best_interface


class BaseInterferogram(ABC):
    default_pipeline_kws = {
        "filter_name": "disk",
        "filter_size": 1 / 3,
        "filter_size_interpretation": "sideband distance",
        "sideband_freq": None,
        "invert_phase": False,
    }

    def __init__(self, data, subtract_mean=True, padding=True, copy=True,
                 **pipeline_kws):
        """Generic class for off-axis hologram data analysis

        Parameters
        ----------
        subtract_mean: bool
            If True, remove the mean of the hologram before performing
            the Fourier transform. This setting is recommended as it
            can reduce artifacts from frequencies around the central
            band.
        padding: bool
            Whether to perform boundary-padding with linear ramp.
            Setting `padding` to `False` increases speed but might
            introduce image distortions such as tilts in the phase
            and amplitude data or dark borders in the amplitude data.
        copy: bool
            Whether to create a copy of the input data.
        pipeline_kws: dict
            Dictionary with defaults for `run_pipeline` as defined in
            `self.default_pipeline_kws`.
        """
        ff_iface = get_best_interface()
        if len(data.shape) == 3:
            # take the first slice (we have alpha or RGB information)
            data = data[:, :, 0]
        #: qpretrieve Fourier transform interface class
        self.fft = ff_iface(data=data,
                            subtract_mean=subtract_mean,
                            padding=padding,
                            copy=copy)
        #: originally computed Fourier transform
        self.fft_origin = self.fft.fft_origin
        #: filtered Fourier data from last run of `run_pipeline`
        self.fft_filtered = self.fft.fft_filtered
        #: last result of `run_pipeline`
        self.field = None
        #: pipeline keyword arguments
        self.pipeline_kws = pipeline_kws

    @property
    def phase(self):
        """Retrieved phase information"""
        if self.field is None:
            self.run_pipeline()
        return np.angle(self.field)

    @property
    def amplitude(self):
        """Retrieved amplitude information"""
        if self.field is None:
            self.run_pipeline()
        return np.abs(self.field)

    def compute_filter_size(self, filter_size, filter_size_interpretation,
                            sideband_freq=None):
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
            fsize = np.sqrt(np.sum(np.array(sideband_freq)**2)) * filter_size
        elif filter_size_interpretation == "frequency index":
            # filter size given in Fourier index (number of Fourier pixels)
            # The user probably does not know that we are padding in
            # Fourier space, so we use the unpadded size and translate it.
            if filter_size <= 0 or filter_size >= self.fft.shape[0] / 2:
                raise ValueError("For frequency index interpretation, "
                                 + "`filter_size` must be between 0 and "
                                 + f"{self.fft.shape[0] / 2}, got "
                                 + f"'{filter_size}'!")
            # convert to frequencies (compatible with fx and fy)
            fsize = filter_size / self.fft.shape[0]
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
        sideband_freq: tuple of floats
            Frequency coordinates of the sideband to use. By default,
            a heuristic search for the sideband is done.
        invert_phase: bool
            Invert the phase data.
        """
