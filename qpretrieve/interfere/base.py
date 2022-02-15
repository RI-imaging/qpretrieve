from abc import ABC, abstractmethod

import numpy as np

from ..fourier import get_best_interface


class BaseInterferogram(ABC):
    def __init__(self, data, subtract_mean=True, copy=True):
        """Generic class for off-axis hologram data analysis"""
        ff_iface = get_best_interface()
        if len(data.shape) == 3:
            # take the first slice (we have alpha or RGB information)
            data = data[:, :, 0]
        #: qpretrieve Fourier transform interface class
        self.fft = ff_iface(data=data,
                            subtract_mean=subtract_mean,
                            padding=True,
                            copy=copy)
        #: originally computed Fourier transform
        self.fft_origin = self.fft.fft_origin
        #: filtered Fourier data from last run of `run_pipeline`
        self.fft_filtered = self.fft.fft_filtered
        #: last result of `run_pipeline`
        self.field = None
        #: hologram pipeline parameters
        self.pipeline_kws = {}

    @property
    def phase(self):
        if self.field is None:
            self.run_pipeline()
        return np.angle(self.field)

    @property
    def amplitude(self):
        if self.field is None:
            self.run_pipeline()
        return np.abs(self.field)

    def compute_filter_size(self, filter_size, filter_size_interpretation,
                            sideband_freq=None):
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
        assert other.__class__.__name__ == self.__class__.__name__
        self.pipeline_kws.clear()
        if not other.pipeline_kws:
            # run default pipeline
            other.run_pipeline()
        self.run_pipeline(**other.pipeline_kws)

    @abstractmethod
    def run_pipeline(self, filter_name="disk", filter_size=1/3,
                     filter_size_interpretation="sideband distance",
                     sideband_freq=None, sideband=+1):
        """Perform pipeline analysis, populating `self.field`"""
