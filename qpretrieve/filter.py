from functools import lru_cache

import numpy as np
from scipy import signal


@lru_cache(maxsize=32)
def get_filter_array(filter_name, filter_size, freq_pos, fft_shape):
    """Create a Fourier filter for holography

    Parameters
    ----------
    filter_name: str
        specifies the filter to use, one of

        - "disk": binary disk with radius `filter_size`
        - "smooth disk": disk with radius `filter_size` convolved
          with a radial gaussian (`sigma=filter_size/5`)
        - "gauss": radial gaussian (`sigma=0.6*filter_size`)
        - "square": binary square with side length `filter_size`
        - "smooth square": square with side length `filter_size`
          convolved with square gaussian (`sigma=filter_size/5`)
        - "tukey": a square tukey window of width `2*filter_size` and
          `alpha=0.1`
    filter_size: float
        Size of the filter in Fourier space. The filter size
        interpreted as a Fourier frequency index ("pixel size")
        and must be between 0 and `max(fft_shape)/2`
    freq_pos: tuple of floats
        The position of the filter in frequency coordinates as
        returned by :func:`nunpy.fft.fftfreq`.
    fft_shape: tuple of int
        The shape of the Fourier transformed image for which the
        filter will be applied. The shape must be squared (two
        identical integers).

    Returns
    -------
    filt_arr: 2d ndarray, boolean of float
        The Fourier-shifted filtering array. For mask images, this
        is a boolean array. For more elaborate filters, this is a
        float array.
    """
    if fft_shape[0] != fft_shape[1]:
        raise ValueError("The Fourier transformed data must have a squared "
                         + f"shape, but the input shape is '{fft_shape}'! "
                         + "Please pad your data properly before FFT.")
    if not (0 < filter_size < max(fft_shape)/2):
        raise ValueError("The filter size cannot exceed more than half of "
                         + "the Fourier space or be negative. Got a filter "
                         + f"size of '{filter_size}' and a shape of "
                         + f"'{fft_shape}'!")
    if not (0
            < min(np.abs(freq_pos))
            <= max(np.abs(freq_pos))
            < max(fft_shape)/2):
        raise ValueError("The frequency position must be within the Fourier "
                         + f"domain. Got '{freq_pos}' and shape "
                         + f"'{fft_shape}'!")

    fx = np.fft.fftshift(np.fft.fftfreq(fft_shape[0])).reshape(-1, 1)
    fy = fx.reshape(1, -1)

    fxc = freq_pos[0] - fx
    fyc = freq_pos[1] - fy

    if filter_name == "disk":
        filter_arr = (fxc ** 2 + fyc ** 2) <= filter_size ** 2
    elif filter_name == "smooth disk":
        sigma = filter_size / 5
        tau = 2 * sigma ** 2
        disk = (fxc ** 2 + fyc ** 2) <= filter_size ** 2
        radsq = fx ** 2 + fy ** 2
        gauss = np.exp(-radsq / tau)
        filter_arr = signal.convolve(gauss, disk, mode="same")
        filter_arr /= filter_arr.max()
    elif filter_name == "gauss":
        sigma = filter_size * .6
        tau = 2 * sigma ** 2
        filter_arr = np.exp(-(fxc ** 2 + fyc ** 2) / tau)
        filter_arr /= filter_arr.max()
    elif filter_name == "square":
        filter_arr = (np.abs(fxc) <= filter_size) \
            * (np.abs(fyc) <= filter_size)
    elif filter_name == "smooth square":
        blur = filter_size / 5
        tau = 2 * blur ** 2
        square = (np.abs(fxc) < filter_size) * (np.abs(fyc) < filter_size)
        gauss = np.exp(-(fy ** 2) / tau) * np.exp(-(fy ** 2) / tau)
        filter_arr = signal.convolve(square, gauss, mode="same")
        filter_arr /= filter_arr.max()
    elif filter_name == "tukey":
        # TODO: avoid the np.roll, instead use the indices directly
        alpha = 0.1
        rsize = int(min(fx.size, fy.size) * filter_size) * 2
        tukey_window_x = signal.tukey(rsize, alpha=alpha).reshape(-1, 1)
        tukey_window_y = signal.tukey(rsize, alpha=alpha).reshape(1, -1)
        tukey = tukey_window_x * tukey_window_y
        base = np.zeros(fft_shape)
        s1 = (np.array(fft_shape) - rsize) // 2
        s2 = (np.array(fft_shape) + rsize) // 2
        base[s1[0]:s2[0], s1[1]:s2[1]] = tukey
        # roll the filter to the peak position
        px = int(freq_pos[0] * fft_shape[0])
        py = int(freq_pos[1] * fft_shape[1])
        filter_arr = np.roll(np.roll(base, px, axis=0), py, axis=1)
    else:
        raise ValueError(f"Unknown filter: {filter_name}")

    return filter_arr
