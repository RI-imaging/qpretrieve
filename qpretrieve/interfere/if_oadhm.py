import numpy as np

from .base import BaseInterferogram


class OffAxisHologram(BaseInterferogram):
    """Generic class for off-axis hologram data analysis"""
    def run_pipeline(self, filter_name="disk", filter_size=1/3,
                     filter_size_interpretation="sideband distance",
                     sideband_freq=None, sideband=+1):
        if sideband_freq is None:
            sideband_freq = find_peak_cosine(self.fft.fft_origin)

        # Get the position of the sideband in frequencies
        if sideband == +1:
            freq_pos = sideband_freq
        elif sideband == -1:
            freq_pos = list(-np.array(sideband_freq))
        else:
            raise ValueError("`sideband` must be +1 or -1!")

        # convert filter_size to frequency coordinates
        fsize = self.compute_filter_size(
            filter_size=filter_size,
            filter_size_interpretation=filter_size_interpretation,
            sideband_freq=sideband_freq)

        self.pipeline_kws = {
            "filter_name": filter_name,
            "filter_size": fsize,
            "filter_size_interpretation": "frequency",
            "sideband_freq": sideband_freq,
            "sideband": sideband
        }

        # perform filtering
        self.field = self.fft.filter(
            filter_name=filter_name, filter_size=fsize, freq_pos=freq_pos)

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
