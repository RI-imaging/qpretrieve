import numpy as np

from .base import BaseInterferogram


class OffAxisHologram(BaseInterferogram):
    """Off-axis hologram data analysis"""
    #: Default OAH pipeline keyword arguments
    default_pipeline_kws = {
        "filter_name": "disk",
        "filter_size": 1 / 3,
        "filter_size_interpretation": "sideband distance",
        "scale_to_filter": False,
        "sideband_freq": None,
        "invert_phase": False,
    }

    @property
    def phase(self):
        """Retrieved phase information"""
        if self._field is None:
            self.run_pipeline()
        if self._phase is None:
            self._phase = np.angle(self._field)
        return self._phase

    @property
    def amplitude(self):
        """Retrieved amplitude information"""
        if self._field is None:
            self.run_pipeline()
            self._amplitude = np.abs(self._field)
        return self._amplitude

    def run_pipeline(self, **pipeline_kws):
        r"""Run OAH analysis pipeline

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
        for key in self.default_pipeline_kws:
            if key not in pipeline_kws:
                pipeline_kws[key] = self.get_pipeline_kw(key)

        if pipeline_kws["sideband_freq"] is None:
            pipeline_kws["sideband_freq"] = find_peak_cosine(
                self.fft.fft_origin)

        # convert filter_size to frequency coordinates
        fsize = self.compute_filter_size(
            filter_size=pipeline_kws["filter_size"],
            filter_size_interpretation=(
                pipeline_kws["filter_size_interpretation"]),
            sideband_freq=pipeline_kws["sideband_freq"])

        # perform filtering
        field = self.fft.filter(
            filter_name=pipeline_kws["filter_name"],
            filter_size=fsize,
            freq_pos=tuple(pipeline_kws["sideband_freq"]),
            scale_to_filter=pipeline_kws["scale_to_filter"])

        if pipeline_kws["invert_phase"]:
            field.imag *= -1

        self._field = field
        self._phase = None
        self._amplitude = None
        self.pipeline_kws.update(pipeline_kws)

        return self.field


def find_peak_cosine(ft_data, copy=True):
    """Find the side band position of a regular off-axis hologram

    The Fourier transform of a cosine function (known as the
    striped fringe pattern in off-axis holography) results in
    two sidebands in Fourier space.

    The hologram is Fourier-transformed and the side band
    is determined by finding the maximum amplitude in
    Fourier space.

    Parameters
    ----------
    ft_data: 2d ndarray
        FFt-shifted Fourier transform of the hologram image
    copy: bool
        copy `ft_data` before modification

    Returns
    -------
    fsx, fsy : tuple of floats
        coordinates of the side band in Fourier space frequencies
    """
    if copy:
        ft_data = ft_data.copy()

    ox, oy = ft_data.shape
    cx = ox // 2
    cy = oy // 2

    minlo = max(int(np.ceil(ox / 42)), 5)
    # remove lower part of Fourier transform to find the peak in the upper
    ft_data[cx - minlo:] = 0

    # remove values around axes
    ft_data[cx - 3:cx + 3, :] = 0
    ft_data[:, cy - 3:cy + 3] = 0

    # find maximum
    am = np.argmax(np.abs(ft_data))
    iy = am % oy
    ix = int((am - iy) / oy)

    fx = np.fft.fftshift(np.fft.fftfreq(ft_data.shape[0]))[ix]
    fy = np.fft.fftshift(np.fft.fftfreq(ft_data.shape[1]))[iy]

    return fx, fy
