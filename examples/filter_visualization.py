"""Filter visualization

This example visualizes the different Fourier filtering masks
available in qpretrieve.
"""
import matplotlib.pylab as plt
import numpy as np
import qpretrieve
from skimage.restoration import unwrap_phase

# load the experimental data
edata = np.load("./data/hologram_cell.npz")

prange = (-1, 5)
frange = (0, 12)

results = {}

for fn in qpretrieve.filter.available_filters:
    holo = qpretrieve.OffAxisHologram(data=edata["data"])
    holo.run_pipeline(
        filter_name=fn,
        # Set the filter size to half the distance
        # between the central band and the sideband.
        filter_size=1/2)
    bg = qpretrieve.OffAxisHologram(data=edata["bg_data"])
    bg.process_like(holo)
    phase = unwrap_phase(holo.phase - bg.phase)
    mask = np.log(1 + np.abs(holo.fft_filtered))
    results[fn] = mask, phase

num_filters = len(results)

# plot the properties of `qpi`
fig = plt.figure(figsize=(8, 22))

for row, name in enumerate(results):
    ax1 = plt.subplot(num_filters, 2, 2*row+1)
    ax1.set_title(name, loc="left")
    ax1.imshow(results[name][0], vmin=frange[0], vmax=frange[1])

    ax2 = plt.subplot(num_filters, 2, 2*row+2)
    map2 = ax2.imshow(results[name][1], cmap="coolwarm",
                      vmin=prange[0], vmax=prange[1])
    plt.colorbar(map2, ax=ax2, fraction=.046, pad=0.02, label="phase [rad]")

    ax1.axis("off")
    ax2.axis("off")

plt.tight_layout()
plt.show()
