"""ROI-based hologram processing vs full-frame processing

This example shows how to speed up off-axis hologram reconstruction
by restricting the FFT processing to regions-of-interest (ROIs) where
objects are located (e.g., inside the channel walls).

It compares the standard full-frame pipeline with ROI processing on
the provided cell hologram, visualises the ROIs, and plots simple
runtime benchmarks.
"""

from time import perf_counter
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import unwrap_phase

import qpretrieve
from qpretrieve._ndarray_backend import xp

# Choose your Backend (numpy will use pyfftw)
backend = "numpy"

# choose your stack size (higher better for cupy comparison)
stack = 20


if backend == "cupy":
    qpretrieve.set_ndarray_backend("cupy")
    fft_interface = qpretrieve.fourier.FFTFilterCupy
else:
    qpretrieve.set_ndarray_backend("numpy")
    fft_interface = qpretrieve.fourier.FFTFilterPyFFTW

# --------------------------------------------------------------------------- #
# Load example data
# --------------------------------------------------------------------------- #
edata = np.load("./data/hologram_cell.npz")
hologram_base = xp.asarray(edata["data"])
bg_base = xp.asarray(edata["bg_data"])
hologram_stack = xp.stack([hologram_base] * stack, axis=0)
bg_stack = xp.stack([bg_base] * stack, axis=0)

# --------------------------------------------------------------------------- #
# Tile the input 3x3 with the hologram only in the center tile
# (use concat for CuPy compatibility)
# --------------------------------------------------------------------------- #
row_bg = xp.concatenate([bg_stack, bg_stack, bg_stack], axis=2)
row_mid = xp.concatenate([bg_stack, hologram_stack, bg_stack], axis=2)
hologram = xp.concatenate([row_bg, row_mid, row_bg], axis=1)
bg_row = xp.concatenate([bg_stack, bg_stack, bg_stack], axis=2)
bg = xp.concatenate([bg_row, bg_row, bg_row], axis=1)
tile_h, tile_w = hologram_base.shape

# Central tile coordinates (middle of the 3x3 grid)
y0, y1 = tile_h, 2 * tile_h
x0, x1 = tile_w, 2 * tile_w
roi_boxes = [(y0, y1, x0, x1)]
channel_mask = xp.zeros_like(hologram, dtype=bool)
channel_mask[:, y0:y1, x0:x1] = True
event_mask = channel_mask.copy()

def to_host(arr):
    """Convert xp array to host numpy for plotting."""
    return xp.asnumpy(arr) if hasattr(xp, "asnumpy") else xp.asarray(arr)

# Reuse one sideband estimate across all runs
def find_sideband():
    holo = qpretrieve.OffAxisHologram(hologram, padding=1,
                                      fft_interface=fft_interface)
    holo.run_pipeline(filter_name="smooth disk")
    return holo.pipeline_kws.get("sideband_freq")

sideband_freq = find_sideband()


def bench(fn, repeat=8):
    start = perf_counter()
    result = None
    for _ in range(repeat):
        result = fn()
    return (perf_counter() - start) / repeat, result


def run_full():
    holo = qpretrieve.OffAxisHologram(hologram, padding=1,
                                      fft_interface=fft_interface)
    holo.run_pipeline(filter_name="smooth disk",
                      sideband_freq=sideband_freq)
    bg_holo = qpretrieve.OffAxisHologram(bg, padding=1,
                                         fft_interface=fft_interface)
    bg_holo.run_pipeline(filter_name="smooth disk",
                         sideband_freq=sideband_freq)
    return holo.phase - bg_holo.phase


def run_roi():
    holo = qpretrieve.OffAxisHologram(hologram, padding=1,
                                      fft_interface=fft_interface)
    holo.run_pipeline_rois(
        roi_boxes=roi_boxes,
        stitch=False,
        filter_name="smooth disk",
        sideband_freq=sideband_freq,
    )
    bg_holo = qpretrieve.OffAxisHologram(bg, padding=1,
                                         fft_interface=fft_interface)
    bg_holo.run_pipeline_rois(
        roi_boxes=roi_boxes,
        stitch=False,
        filter_name="smooth disk",
        sideband_freq=sideband_freq,
    )
    return holo.phase - bg_holo.phase


def run_roi_stitch():
    holo = qpretrieve.OffAxisHologram(hologram, padding=1,
                                      fft_interface=fft_interface)
    holo.run_pipeline_rois(
        roi_boxes=roi_boxes,
        stitch=True,
        filter_name="smooth disk",
        sideband_freq=sideband_freq,
    )
    bg_holo = qpretrieve.OffAxisHologram(bg, padding=1,
                                         fft_interface=fft_interface)
    bg_holo.run_pipeline_rois(
        roi_boxes=roi_boxes,
        stitch=True,
        filter_name="smooth disk",
        sideband_freq=sideband_freq,
    )
    return holo.phase - bg_holo.phase


# run pyfftw warmups
_ = bench(run_full)
_ = bench(run_roi)
_ = bench(run_roi_stitch)
# now actually take the times
full_time, phase_full = bench(run_full)
roi_time, phase_roi = bench(run_roi)
roi_stitch_time, phase_roi_stitch = bench(run_roi_stitch)
phase_full = to_host(phase_full)
phase_roi = to_host(phase_roi)
phase_roi_stitch = to_host(phase_roi_stitch)
phase_full_disp = phase_full[0]
phase_roi_disp = phase_roi[0]
phase_roi_stitch_disp = phase_roi_stitch[0]
phase_full_tile = phase_full_disp[y0:y1, x0:x1]
# Unwrapped (not timed)
phase_full_unwrap = unwrap_phase(phase_full_disp)
phase_roi_unwrap = unwrap_phase(phase_roi_disp)
phase_roi_stitch_unwrap = unwrap_phase(phase_roi_stitch_disp)

# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
fig, axes = plt.subplots(2, 3, figsize=(12, 7))

ax = axes[0, 0]
ax.set_title("Hologram")
ax.imshow(to_host(hologram[0]), cmap="gray")
ax.axis("off")

ax = axes[0, 1]
ax.set_title("Channel mask & ROI")
mask_vis = xp.zeros_like(hologram, dtype=float)
mask_vis[channel_mask] = 0.4
mask_vis[event_mask] = 1.0
ax.imshow(to_host(mask_vis[0]), cmap="magma")
for (y0, y1, x0, x1) in roi_boxes:
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                               ec="cyan", fc="none", lw=2))
ax.axis("off")

ax = axes[0, 2]
ax.set_title(f"Runtime {fft_interface.__name__} (unwrapping excl.)")
ax.bar(["full", "roi", "roi+stitch"],
       [full_time, roi_time, roi_stitch_time],
       color=["tab:blue", "tab:green", "tab:orange"])
ax.set_ylabel("seconds (lower is faster)")
ax.annotate("Unwrapping done after timing", xy=(0.5, -0.12),
            xycoords="axes fraction", ha="center", va="top", fontsize=9)

ax = axes[1, 0]
ax.set_title("Phase unwrapped (full 3x3)")
im = ax.imshow(phase_full_unwrap, cmap="coolwarm")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                           ec="cyan", fc="none", lw=2))

ax = axes[1, 1]
ax.set_title("Phase unwrapped (ROI only)")
im = ax.imshow(phase_roi_unwrap, cmap="coolwarm")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

ax = axes[1, 2]
ax.set_title("Phase unwrapped (ROI stitched)")
im = ax.imshow(phase_roi_stitch_unwrap, cmap="coolwarm")
ax.axis("off")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig("roi_hologram.png", dpi=150)
plt.show()
