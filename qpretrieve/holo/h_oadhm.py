import numpy as np

from ..fourier import get_best_interface


class OffAxisHologram:
    def __init__(self, data, subtract_mean=True, copy=True,
                 sideband_freq=None):
        """Generic class for off-axis hologram data analysis"""
        ff_iface = get_best_interface()
        self.fft = ff_iface(data=data,
                            subtract_mean=subtract_mean,
                            padding=True,
                            copy=copy)
        self.fft_origin = self.fft.fft_origin

        if sideband_freq is None:
            self.sideband_freq = find_peak_cosine(self.fft.fft_origin)
        else:
            self.sideband_freq = sideband_freq

        self.field = None

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

    def run_pipeline(self, sideband=+1, filter_name="disk", filter_size=1/3,
                     filter_size_interpretation="sideband distance"):
        # Get the position of the sideband in frequencies
        if sideband == +1:
            freq_pos = self.sideband_freq
        if sideband == -1:
            freq_pos = list(-np.array(self.sideband_freq))
        else:
            raise ValueError("`sideband` must be +1 or -1!")

        if filter_size_interpretation == "sideband distance":
            # filter size based on distance b/w central band and sideband
            if filter_size <= 0 or filter_size >= 1:
                raise ValueError("For sideband distance interpretation, "
                                 "`filter_size` must be between 0 and 1; "
                                 f"got '{filter_size}'!")
            fsize = np.sqrt(np.sum(filter_size**2)) * filter_size
        elif filter_size_interpretation == "frequency":
            # convert frequency to frequency index
            # We always have padded Fourier data with sizes of order 2.
            fsize = filter_size
        elif filter_size_interpretation == "frequency index":
            # filter size given in Fourier index (number of Fourier pixels)
            # The user probably does not know that we are padding in
            # Fourier space, so we use the unpadded size and translate it.
            if filter_size <= 0 or filter_size >= self.fft.shape[0] / 2:
                raise ValueError("For frequency index interpretation, "
                                 "`filter_size` must be between 0 and "
                                 f"{self.fft.shape[0]}, got '{filter_size}'!")
            # convert to frequencies (compatible with fx and fy)
            fsize = filter_size / self.fft.shape[0]
        else:
            raise ValueError("Invalid value for `filter_size_interpretation`: "
                             + f"'{filter_size_interpretation}'")

        # perform filtering
        self.field = self.fft.filter(
            filter_name=filter_name, filter_size=fsize, freq_pos=freq_pos)
        self.fft_filtered = self.fft.fft_filtered

        return self.field


def find_peak_cosine(ft_data, copy=True):
    """Find the side band position of a regular fringe hologram

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
