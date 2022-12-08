"""Adequate resolution-scaling via cropping in Fourier space

Applying filters in Fourier space usually means setting a larger
fraction of the data in Fourier space to zero. This can considerably
slow down your analysis, since the inverse Fouier transform is
taking into account a lot of unused frequencies.

By cropping the Fourier domain, the inverse Fourier transform will
be faster. The result is an image with fewer pixels (lower resolution),
that contains the same information as the unscaled (not cropped in Fourier
space) image. There are possibly two disadvantages of performing this
cropping operation:

1. The resolution of the output image (pixel size) is not the same as
   that of the input (interference) image. Thus, you will have to adapt
   your colocalization scheme.
2. If you are using a filter with a non-binary mask (e.g. gauss), then
   you will lose (little) information when cropping in Fourier space.

What happens when you set `filter_name="smooth disk"`? Cropping with
`scale_to_filter=True` is too narrow, because now there are higher
frequencies contributing to the final image. To include them, you can
set `scale_to_filter` to a float, e.g. `scale_to_filter=1.2`.
"""
import matplotlib.pylab as plt
import numpy as np
import qpretrieve
from skimage.restoration import unwrap_phase

# load the experimental data
edata = np.load("./data/hologram_cell.npz")

results = []

holo = qpretrieve.OffAxisHologram(data=edata["data"])
bg = qpretrieve.OffAxisHologram(data=edata["bg_data"])
ft_orig = np.log(1 + np.abs(holo.fft_origin), dtype=float)

holo.run_pipeline(filter_name="disk", filter_size=1/2,
                  scale_to_filter=False)
bg.process_like(holo)
phase_highres = unwrap_phase(holo.phase - bg.phase)
ft_highres = np.log(1 + np.abs(holo.fft.fft_used), dtype=float)

holo.run_pipeline(filter_name="disk", filter_size=1/2,
                  scale_to_filter=True)
bg.process_like(holo)
phase_scalerad = unwrap_phase(holo.phase - bg.phase)
ft_scalerad = np.log(1 + np.abs(holo.fft.fft_used), dtype=float)

# plot the intermediate steps of the analysis pipeline
fig = plt.figure(figsize=(8, 10))

ax1 = plt.subplot(321, title="cell hologram")
map1 = ax1.imshow(edata["data"], interpolation="bicubic", cmap="gray")
plt.colorbar(map1, ax=ax1, fraction=.046, pad=0.04)

ax2 = plt.subplot(322, title="Fourier transform")
map2 = ax2.imshow(ft_orig, cmap="viridis")
plt.colorbar(map2, ax=ax2, fraction=.046, pad=0.04)

ax3 = plt.subplot(323, title="Full, filtered Fourier transform")
map3 = ax3.imshow(ft_highres, cmap="viridis")
plt.colorbar(map3, ax=ax3, fraction=.046, pad=0.04)

ax4 = plt.subplot(324, title="High-res phase [rad]")
map4 = ax4.imshow(phase_highres, cmap="coolwarm")
plt.colorbar(map4, ax=ax4, fraction=.046, pad=0.04)

ax5 = plt.subplot(325, title="Cropped, filtered Fourier transform")
map5 = ax5.imshow(ft_scalerad, cmap="viridis")
plt.colorbar(map5, ax=ax5, fraction=.046, pad=0.04)

ax6 = plt.subplot(326, title="Low-res, same-information phase [rad]")
map6 = ax6.imshow(phase_scalerad, cmap="coolwarm")
plt.colorbar(map6, ax=ax6, fraction=.046, pad=0.04)

# disable axes
[ax.axis("off") for ax in [ax1, ax2, ax5, ax6]]

plt.tight_layout()
plt.show()
