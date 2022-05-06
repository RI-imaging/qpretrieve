import warnings

import numpy as np
import scipy
from skimage.restoration import unwrap_phase

from .base import BaseInterferogram
from ..fourier import get_best_interface


class QLSInterferogram(BaseInterferogram):
    """Generic class for quadri-wave lateral shearing holograms"""
    default_pipeline_kws = {
        "filter_name": "square",
        "filter_size": 0.5,
        "filter_size_interpretation": "sideband distance",
        "sideband_freq": None,
        "invert_phase": False,
        "wavelength": None,
        "qlsi_pitch_term": None,
    }

    def __init__(self, data, reference=None, *args, **kwargs):
        super(QLSInterferogram, self).__init__(data, *args, **kwargs)
        ff_iface = get_best_interface()

        if reference is not None:
            self.fft_ref = ff_iface(data=reference,
                                    subtract_mean=self.fft.subtract_mean,
                                    padding=self.fft.padding)

        self.wavefront = None
        self._phase = None
        self._amplitude = None
        self._field = None

    @property
    def amplitude(self):
        if self._amplitude is None:
            self.run_pipeline()
        return self._amplitude

    @property
    def field(self):
        if self._field is None:
            self._field = self.amplitude * np.exp(1j*2*np.pi*self.phase)
        return self._field

    @property
    def phase(self):
        if self._phase is None:
            self.run_pipeline()
        return self._phase

    def run_pipeline(self, **pipeline_kws):
        """Run QLSI analysis pipeline

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
        wavelength: float
            Wavelength to convert from the wavefront in meters to radians.
        qlsi_pitch_term: float
            Scaling term converting the integrated gradient image to
            the unit meters. This term is dependent on the lattice
            constant of the grating :math:`L`, the distance between the
            grating and the camera sensor :math:`d` and the physical camera
            pixel width :math:`a` according to

            .. math::

                \text{pitch_term} = \frac{La}{d}

            For the case where the lattice constant is four times the
            pixel width, this simplifies to :math:`4a^2/d`. Note
            that for a relay-lens system (grating not directly attached
            to the sensor) this factor is wavelength dependent due to
            chromatic aberrations introduced by the lenses. For
            gratings-on-a-camera configurations (e.g. Phasics SID4Bio),
            this is a device-specific quantity which has to be determined
            only once. E.g. for our SID4Bio camera, this value is
            0.01887711 µm (1.87711e-08 m).
        """
        for key in self.default_pipeline_kws:
            if key not in pipeline_kws:
                pipeline_kws[key] = self.get_pipeline_kw(key)

        if pipeline_kws["sideband_freq"] is None:
            pipeline_kws["sideband_freq"] = find_peaks_qlsi(
                self.fft.fft_origin)

        # convert filter_size to frequency coordinates
        fsize = self.compute_filter_size(
            filter_size=pipeline_kws["filter_size"],
            filter_size_interpretation=(
                pipeline_kws["filter_size_interpretation"]),
            sideband_freq=pipeline_kws["sideband_freq"])

        # get pitch ratio
        qlsi_pitch_term = pipeline_kws["qlsi_pitch_term"]
        if qlsi_pitch_term is None:
            warnings.warn("No `qlsi_pitch_term` specified! Your phase data "
                          "is only qualitative, not quantitatively correct!")
            qlsi_pitch_term = 1

        # get pitch ratio
        wavelength = pipeline_kws["wavelength"]
        if wavelength is None:
            warnings.warn("No `wavelength` specified! Your phase data "
                          "is only qualitative, not quantitatively correct!")
            wavelength = 1

        # Obtain Hx and Hy by filtering the Fourier transform at the
        # two frequencies and performing an inverse Fourier transform.
        fx, fy = pipeline_kws["sideband_freq"]
        hx = self.fft.filter(filter_name=pipeline_kws["filter_name"],
                             filter_size=fsize,
                             freq_pos=(fx, fy))
        hy = self.fft.filter(filter_name=pipeline_kws["filter_name"],
                             filter_size=fsize,
                             freq_pos=(-fy, fx))

        # Subtract the reference from the gradient data
        if self.fft_ref is not None:
            hbx = self.fft_ref.filter(filter_name=pipeline_kws["filter_name"],
                                      filter_size=fsize,
                                      freq_pos=(fx, fy))
            hby = self.fft_ref.filter(filter_name=pipeline_kws["filter_name"],
                                      filter_size=fsize,
                                      freq_pos=(-fy, fx))
            hx /= hbx
            hy /= hby

        # Obtain the phase gradients in x and y by taking the argument
        # of Hx and Hy.
        px = unwrap_phase(np.angle(hx))
        py = unwrap_phase(np.angle(hy))

        # Determine the angle by which we have to rotate the gradients in
        # order for them to be aligned with x and y. This angle is defined
        # by the frequency positions.
        angle = np.arctan2(fy, fx)

        # Pad the gradient information so that we can rotate with cropping
        # (keeping the image shape the same).
        # TODO: Make padding dependent on rotation angle to save time?
        sx, sy = self.fft_origin.shape
        gradpad1 = np.pad(px, ((sx // 2, sx // 2), (sy // 2, sy // 2)),
                          mode="median")
        gradpad2 = np.pad(py, ((sx // 2, sx // 2), (sy // 2, sy // 2)),
                          mode="median")

        # Perform rotation of the gradients.
        rotated1 = rotate_noreshape(gradpad1, -angle)
        rotated2 = rotate_noreshape(gradpad2, -angle)

        # Retrieve the wavefront by integrating the vectorial components
        # (integrate the total differential). This magical approach
        # puts the x gradient in the real and the y gradient in the imaginary
        # part.
        ff_iface = get_best_interface()
        rfft = ff_iface(data=rotated1 + 1j * rotated2,
                        subtract_mean=False,
                        padding=False,
                        copy=False)
        # Compute the frequencies that correspond to the frequencies of the
        # Fourier-transformed image.
        fx = np.fft.fftfreq(rfft.shape[0]).reshape(-1, 1)
        fy = np.fft.fftfreq(rfft.shape[1]).reshape(1, -1)
        fxy = -2*np.pi*1j * (fx + 1j*fy)
        fxy[0, 0] = 1

        # The wavefront is the real part of the inverse Fourier transform
        # of the filtered (divided by frequencies) data.
        wfr = rfft._ifft(np.fft.ifftshift(rfft.fft_origin)/fxy).real

        # Rotate the wavefront back and crop it so that the FOV matches
        # the input data.
        raw_wavefront = rotate_noreshape(wfr,
                                         angle)[sx//2:-sx//2, sy//2:-sy//2]
        # Multiply by qlsi pitch term to get a quantitative wavefront.
        raw_wavefront *= qlsi_pitch_term

        self._phase = raw_wavefront / wavelength * 2 * np.pi
        # TODO: Is adding these abs values really the amplitude?
        amp = np.abs(hx) + np.abs(hy)

        self._amplitude = amp

        self.pipeline_kws.update(pipeline_kws)

        self.wavefront = raw_wavefront

        return raw_wavefront


def find_peaks_qlsi(ft_data, periodicity=4, copy=True):
    """Find the two peaks in Fourier space for the x and y gradient

    Parameters
    ----------
    ft_data: 2d complex ndarray
        FFT-shifted Fourier transform of the QLSI image
    periodicity: float
        Grid size of the QLSI image. For the Phasics SID4Bio
        camera, this is `4` (i.e. the peak-to-peak distance of
        the individual foci in the QLSI image is four pixels)
    copy: bool
        Set to False to perform operations in-place.

    Returns
    -------
    (f1x, f1y): tuple of floats
        Coordinates of the first gradient peak in frequency
        coordinates.
    (f2x, f2y): tuple of floats
        Coordinates of the second gradient peak in frequency
        coordinates.

    TODO
    ----
    At some point it might be necessary to add an `angle` keyword
    argument that gives the algorithm a hint about te rotation of
    the QLSI grid. Currently, peak detection is only done in the
    lower half of `ft_data`. If the peaks are exactly aligned with
    the pixel grid, then the current approach might not work. Also,
    setting `angle=np.pi` would be equivalent to setting sideband
    to -1 in holo.py (would be a nice feature).
    """
    if copy:
        ft_data = ft_data.copy()

    ox, oy = ft_data.shape
    cx = ox // 2
    cy = oy // 2

    # We only look at the lower right image. This corresponds to using
    # only one sideband in (as in holo.py).
    minlo = max(int(np.ceil(ox / 42)), 5)
    ft_data[cx - minlo:] = 0

    # remove values around axes
    ft_data[cx - 3:cx + 3, :] = 0
    ft_data[:, cy - 3:cy + 3] = 0

    # circular bandpass according to periodicity
    fx = np.fft.fftshift(np.fft.fftfreq(ft_data.shape[0])).reshape(-1, 1)
    fy = np.fft.fftshift(np.fft.fftfreq(ft_data.shape[1])).reshape(1, -1)
    frmask1 = np.sqrt(fx**2 + fy**2) > 1/(periodicity*.8)
    frmask2 = np.sqrt(fx ** 2 + fy ** 2) < 1 / (periodicity * 1.2)
    ft_data[np.logical_or(frmask1, frmask2)] = 0

    # find the peak in the left part
    am1 = np.argmax(np.abs(ft_data*(fy < 0)))
    i1y = am1 % oy
    i1x = int((am1 - i1y) / oy)

    return fx[i1x, 0], fy[0, i1y]


def rotate_noreshape(arr, angle, mode="mirror", reshape=False):
    return scipy.ndimage.rotate(
        arr,  # input
        angle=np.rad2deg(angle),  # angle
        reshape=reshape,  # reshape
        order=0,  # order
        mode=mode,  # mode
        prefilter=False,
        cval=0)
