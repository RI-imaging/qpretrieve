from __future__ import annotations

from .._ndarray_backend import xp
from ..fourier import FourierFieldData
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
        "output_domain": "spatial",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fourier_field_data = None

    @property
    def field(self) -> xp.ndarray:
        """Retrieved complex field information."""
        if self._field is None:
            if self._fourier_field_data is not None:
                self.compute_field()
            else:
                self.run_pipeline(output_domain="spatial")
        return self._field

    @property
    def phase(self) -> xp.ndarray:
        """Retrieved phase information"""
        if self._field is None:
            if self._fourier_field_data is not None:
                self.compute_field()
            else:
                self.run_pipeline(output_domain="spatial")
        if self._phase is None:
            self._phase = xp.angle(self._field)
        return self._phase

    @property
    def amplitude(self) -> xp.ndarray:
        """Retrieved amplitude information"""
        if self._field is None:
            if self._fourier_field_data is not None:
                self.compute_field()
            else:
                self.run_pipeline(output_domain="spatial")
        if self._amplitude is None:
            self._amplitude = xp.abs(self._field)
        return self._amplitude

    def compute_field(self,
                      propagated_fft: xp.ndarray | None = None) -> xp.ndarray:
        """Compute the field using the current pipeline settings.

        If the field was skipped previously with ``output_domain='fourier'``,
        this will reuse the cached Fourier intermediates instead of
        recomputing the full filter path.

        Parameters
        ----------
        propagated_fft: ndarray, optional
            Fourier-domain field after external propagation. If omitted,
            the stored qpretrieve Fourier artifact is computed directly.

        """
        if self._field is None:
            if self._fourier_field_data is None:
                self.run_pipeline(output_domain="spatial")
            else:
                self._field = self._fourier_field_data.finalize(
                    propagated_fft=propagated_fft)
        return self._field

    def run_pipeline(self, output_domain: str = "spatial",
                     **pipeline_kws) -> xp.ndarray | FourierFieldData:
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
            If set to "physical radius", the base radius is
            :math:`r \approx n dx NA / \lambda` in Fourier pixels. In this
            mode `filter_size` is a scaling factor (`1.0` means direct
            physical radius).
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
            If you pass a 3D array, the first hologram is used to
            determine the sideband frequencies.
        pixel_size: float
            Sensor pixel size `dx` in meters for physical-radius mode.
        numerical_aperture: float
            Collection NA for physical-radius mode.
        wavelength: float
            Illumination wavelength in meters for physical-radius mode.
        invert_phase: bool
            Invert the phase data.
        output_domain: str
            Either ``"spatial"`` or ``"fourier"``. Spatial returns the
            field, Fourier returns a :class:`FourierFieldData`.

            .. versionadded:: 0.7.0
        """
        pipeline_kws["output_domain"] = output_domain
        for key in self.default_pipeline_kws:
            if key not in pipeline_kws:
                pipeline_kws[key] = self.get_pipeline_kw(key)

        if pipeline_kws["sideband_freq"] is None:
            pipeline_kws["sideband_freq"] = find_peak_cosine(
                self.fft.fft_origin[0])

        # convert filter_size to frequency coordinates
        fsize = self.compute_filter_size(
            filter_size=pipeline_kws["filter_size"],
            filter_size_interpretation=(
                pipeline_kws["filter_size_interpretation"]),
            sideband_freq=pipeline_kws["sideband_freq"],
            pixel_size=pipeline_kws.get("pixel_size"),
            numerical_aperture=pipeline_kws.get("numerical_aperture"),
            wavelength=pipeline_kws.get("wavelength"))

        # perform filtering
        filter_size = float(fsize)
        freq_pos = tuple(float(x) for x in pipeline_kws["sideband_freq"])

        if pipeline_kws["output_domain"] == "spatial":
            # legacy pipeline
            field = self.fft.filter(
                filter_name=pipeline_kws["filter_name"],
                filter_size=filter_size,
                freq_pos=freq_pos,
                scale_to_filter=pipeline_kws["scale_to_filter"],
                output_domain="spatial")

            if pipeline_kws["invert_phase"]:
                field.imag *= -1

            self._field = field
            self._fourier_field_data = None
        else:
            # direct fourier pipeline
            artifact = self.fft.filter(
                filter_name=pipeline_kws["filter_name"],
                filter_size=filter_size,
                freq_pos=freq_pos,
                scale_to_filter=pipeline_kws["scale_to_filter"],
                output_domain="fourier")
            self._field = None
            self._fourier_field_data = artifact

        self._phase = None
        self._amplitude = None
        self.pipeline_kws.update(pipeline_kws)

        if pipeline_kws["output_domain"] == "spatial":
            return self._field
        else:
            return self._fourier_field_data


def find_peak_cosine(
        ft_data: xp.ndarray, copy: bool = True) -> tuple[float, float]:
    """Find the side band position of a 2d regular off-axis hologram

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

    minlo = max(int(xp.ceil(ox / 42)), 5)
    # remove lower part of Fourier transform to find the peak in the upper
    ft_data[cx - minlo:] = 0

    # remove values around axes
    ft_data[cx - 3:cx + 3, :] = 0
    ft_data[:, cy - 3:cy + 3] = 0

    # find maximum
    am = xp.argmax(xp.abs(ft_data))
    iy = am % oy
    ix = int((am - iy) / oy)

    fx = xp.fft.fftshift(xp.fft.fftfreq(ft_data.shape[0]))[ix]
    fy = xp.fft.fftshift(xp.fft.fftfreq(ft_data.shape[1]))[iy]

    return fx, fy
