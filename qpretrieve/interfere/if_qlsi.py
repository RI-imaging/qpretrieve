from functools import lru_cache

import numpy as np
import scipy
from skimage.restoration import unwrap_phase

from .base import BaseInterferogram
from ..fourier import get_best_interface


class QLSReference:
    def __init__(self, reference):
        ff_iface = get_best_interface()
        self.fft = ff_iface(data=reference,
                            subtract_mean=True,
                            padding=True,
                            copy=True)

    @lru_cache(maxsize=32)
    def get_gradients(self, filter_name, filter_size, sideband_freq):
        fx, fy = sideband_freq
        hx = self.fft.filter(filter_name=filter_name,
                             filter_size=filter_size,
                             freq_pos=(fx, fy))
        px = unwrap_phase(np.angle(hx))
        hy = self.fft.filter(filter_name=filter_name,
                             filter_size=filter_size,
                             freq_pos=(-fy, fx))
        py = unwrap_phase(np.angle(hy))
        self.amplitude = np.abs(hx) + np.abs(hy)
        return px, py


class QLSInterferogram(BaseInterferogram):
    """Generic class for quadri-wave lateral shearing holograms"""
    default_pipeline_kws = {
        "filter_name": "square",
        "filter_size": 400,
        "filter_size_interpretation": "frequency index",
        "sideband_freq": None,
        "invert_phase": False,
    }

    def __init__(self, data, reference=None, *args, **kwargs):
        super(QLSInterferogram, self).__init__(data, *args, **kwargs)
        if reference is not None:
            self.set_reference(reference)
        else:
            self.reference = None
        self._phase = None
        self._amplitude = None

    @property
    def amplitude(self):
        if self._amplitude is None:
            self.run_pipeline()
        return self._amplitude

    @property
    def phase(self):
        if self._phase is None:
            self.run_pipeline()
        return self._phase

    def run_pipeline(self, **pipeline_kws):
        for key in self.default_pipeline_kws:
            if key not in pipeline_kws:
                pipeline_kws[key] = self.get_pipeline_kw(key)

        if pipeline_kws["sideband_freq"] is None:
            pipeline_kws["filter_name"], = find_peaks_qlsi(
                self.fft.fft_origin)

        # convert filter_size to frequency coordinates
        fsize = self.compute_filter_size(
            filter_size=pipeline_kws["filter_size"],
            filter_size_interpretation=(
                pipeline_kws["filter_size_interpretation"]),
            sideband_freq=pipeline_kws["sideband_freq"])

        fx, fy = pipeline_kws["sideband_freq"]
        hx = self.fft.filter(filter_name=pipeline_kws["filter_name"],
                             filter_size=fsize,
                             freq_pos=(fx, fy))
        hy = self.fft.filter(filter_name=pipeline_kws["filter_name"],
                             filter_size=fsize,
                             freq_pos=(-fy, fx))

        px = unwrap_phase(np.angle(hx))
        py = unwrap_phase(np.angle(hy))

        pbgx, pbgy = self.reference.get_gradients(
            filter_name=pipeline_kws["filter_name"],
            filter_size=fsize,
            sideband_freq=pipeline_kws["sideband_freq"])

        px -= pbgx
        py -= pbgy

        angle = np.arctan2(fy, fx)

        sx, sy = self.fft_origin.shape
        gradpad1 = np.pad(px, ((sx // 2, sx // 2), (sy // 2, sy // 2)),
                          mode="median")
        gradpad2 = np.pad(py, ((sx // 2, sx // 2), (sy // 2, sy // 2)),
                          mode="median")

        rotated1 = rotate_noreshape(gradpad1, -angle)
        rotated2 = rotate_noreshape(gradpad2, -angle)
        ff_iface = get_best_interface()

        # retrieve scalar field by integrating the vectorial components
        # (integrate the total differential)
        rfft = ff_iface(data=rotated1 + 1j * rotated2,
                        subtract_mean=False,
                        padding=False,
                        copy=False)
        fx = np.fft.fftfreq(rfft.shape[0]).reshape(-1, 1)
        fy = np.fft.fftfreq(rfft.shape[1]).reshape(1, -1)
        fxy = -2*np.pi*1j * (fx + 1j*fy)
        fxy[0, 0] = 1

        phaser = rfft._ifft(np.fft.ifftshift(rfft.fft_origin)/fxy).real

        self._phase = rotate_noreshape(phaser,
                                       angle)[sx//2:-sx//2, sy//2:-sy//2]
        amp = np.abs(hx) + np.abs(hy)
        self._amplitude = amp / self.reference.amplitude

        self.field = self._amplitude * np.exp(1j*2*np.pi*self._phase)

        self.pipeline_kws.update(pipeline_kws)

        return self.field

    def set_reference(self, reference):
        self.reference = QLSReference(reference)


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
    return scipy.ndimage.interpolation.rotate(
        arr,  # input
        angle=np.rad2deg(angle),  # angle
        reshape=reshape,  # reshape
        order=0,  # order
        mode=mode,  # mode
        prefilter=False,
        cval=0)
