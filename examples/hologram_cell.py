"""Digital hologram of a single cell

This example illustrates how qpretrieve can be used to analyze
digital holograms. The hologram of a single myeloid leukemia
cell (HL60) was recorded using off-axis digital holographic microscopy
(DHM). Because the phase-retrieval method used in DHM is based on the
*discrete* Fourier transform, there always is a residual background
phase tilt which must be removed for further image analysis.
The setup used for recording these data is described in reference
:cite:`Schuermann2015`.

Note that the fringe pattern in this particular example is over-sampled
in real space, which is why the sidebands are not properly separated
in Fourier space. Thus, the filter in Fourier space is very small
which results in a very low effective resolution in the reconstructed
phase.
"""
import matplotlib.pylab as plt
import numpy as np
import qpretrieve
from skimage.restoration import unwrap_phase

# load the experimental data
edata = np.load("./data/hologram_cell.npz")

holo = qpretrieve.OffAxisHologram(data=edata["data"])
holo.run_pipeline(
    # For this hologram, the "smooth disk"
    # filter yields the best trade-off
    # between interference from the central
    # band and image resolution.
    filter_name="smooth disk",
    # Set the filter size to half the distance
    # between the central band and the sideband.
    filter_size=1/2)
bg = qpretrieve.OffAxisHologram(data=edata["bg_data"])
bg.process_like(holo)

phase = holo.phase - bg.phase

# plot the properties of `qpi`
fig = plt.figure(figsize=(8, 10))

ax1 = plt.subplot(321, title="cell hologram")
map1 = ax1.imshow(edata["data"], interpolation="bicubic", cmap="gray")
plt.colorbar(map1, ax=ax1, fraction=.046, pad=0.04)

ax2 = plt.subplot(322, title="bg hologram")
map2 = ax2.imshow(edata["bg_data"], interpolation="bicubic", cmap="gray")
plt.colorbar(map2, ax=ax2, fraction=.046, pad=0.04)

ax3 = plt.subplot(323, title="Fourier transform of cell")
map3 = ax3.imshow(np.log(1 + np.abs(holo.fft_origin)), cmap="viridis")
plt.colorbar(map3, ax=ax3, fraction=.046, pad=0.04)

ax4 = plt.subplot(324, title="filtered Fourier transform of cell")
map4 = ax4.imshow(np.log(1 + np.abs(holo.fft_filtered)), cmap="viridis")
plt.colorbar(map4, ax=ax4, fraction=.046, pad=0.04)

ax5 = plt.subplot(325, title="retrieved phase [rad]")
map5 = ax5.imshow(phase, cmap="coolwarm")
plt.colorbar(map5, ax=ax5, fraction=.046, pad=0.04)

ax6 = plt.subplot(326, title="unwrapped phase [rad]")
map6 = ax6.imshow(unwrap_phase(phase), cmap="coolwarm")
plt.colorbar(map6, ax=ax6, fraction=.046, pad=0.04)

# disable axes
[ax.axis("off") for ax in [ax1, ax2, ax3, ax4, ax5, ax6]]

plt.tight_layout()
plt.show()
