"""Spatial vs. Direct (Fourier) Pipeline with/without padding

Combined qpretrieve + nrefocus pipeline for the spatial
vs direct Fourier pipeline.

Two padding cases show the difference in output:

- padding=False
    - the two pipelines are numerically identical. However, without
      padding more phase discontinuities occur.
- padding=True
    - The pipelines output slightly difference results.
      The spatial path tells nrefocus to
      re-pad the already-finalized field, while the fourier
      path propagates the padded Fourier data
      directly without a second padding step. These are
      physically different operations, so the outputs will not match.

This script just demonstrates the spatial vs. fourier pipeline,
we do not do any background subtraction or phase unwrapping.
"""

import matplotlib.pyplot as plt
import numpy as np
import qpretrieve
import nrefocus

qpretrieve.set_ndarray_backend("numpy")
nrefocus.set_ndarray_backend("numpy")

edata = np.load("./data/hologram_cell.npz")
hologram = edata["data"].copy()
# crop to the smaller side
_s = min(hologram.shape)
hologram_sq = hologram[:_s, :_s]
propagation_kwargs = dict(d=1.5, nm=1.533, res=8.25, method="fresnel")


def _spatial_pipeline(hologram, padding_qpr, padding_nrf):
    holo = qpretrieve.OffAxisHologram(hologram, padding=padding_qpr)
    field = holo.run_pipeline(output_domain="spatial")
    return nrefocus.refocus(field=field, padding=padding_nrf,
                            **propagation_kwargs)


def _fourier_pipeline(hologram, padding_qpr, padding_nrf):
    holo = qpretrieve.OffAxisHologram(hologram, padding=padding_qpr)
    artifact = holo.run_pipeline(output_domain="fourier")
    return nrefocus.refocus(field=artifact, padding=padding_nrf,
                            **propagation_kwargs)


# spatial pipeline - no padding (square crop required) - pipelines match
pad_zero = 0
field_raw_nopad = qpretrieve.OffAxisHologram(
    hologram_sq, padding=pad_zero).run_pipeline(output_domain="spatial")
field_spatial_nopad = _spatial_pipeline(hologram_sq, pad_zero, pad_zero)
field_fourier_nopad = _fourier_pipeline(hologram_sq, pad_zero, pad_zero)

assert np.allclose(field_spatial_nopad, field_fourier_nopad, atol=1e-10), (
    f"padding={pad_zero} don't match when they should.")
max_diff = np.abs(field_spatial_nopad - field_fourier_nopad).max()
print(f"padding={pad_zero}: spatial − fourier = {max_diff:.4g}")

pad_one = 1
# new direct pipeline - with padding - pipelines do not match (expected)
field_raw_pad = qpretrieve.OffAxisHologram(
    hologram, padding=pad_one).run_pipeline(output_domain="spatial")
field_spatial_pad = _spatial_pipeline(hologram, pad_one, pad_one)
field_fourier_pad = _fourier_pipeline(hologram, pad_one, pad_one)

assert not np.allclose(field_spatial_pad, field_fourier_pad, atol=1e-10)
max_diff = np.abs(field_spatial_pad - field_fourier_pad).max()
print(f"padding={pad_one}: spatial − fourier = {max_diff:.4g}  (expected)")

field_spatial_pad_cropped = field_spatial_pad[:, 10:190, 10:190]
field_fourier_pad_cropped = field_fourier_pad[:, 10:190, 10:190]
assert not np.allclose(field_spatial_pad_cropped, field_fourier_pad_cropped,
                       atol=1e-10)
max_diff = np.abs(field_spatial_pad_cropped - field_fourier_pad_cropped).max()
print(f"padding={pad_one}, central crop: spatial − fourier = {max_diff:.4g}")


def _phase(field):
    return np.angle(field[0])


diff_pad = _phase(field_spatial_pad) - _phase(field_fourier_pad)
diff_nopad = _phase(field_spatial_nopad) - _phase(field_fourier_nopad)

panel_titles = [
    ("raw phase", "raw phase"),
    ("spatial propagation", "spatial propagation"),
    ("fourier propagation", "fourier propagation"),
    ("difference: spatial − fourier  ✓ match",
     "difference: spatial − fourier  ✗ differ"),
]
rows = [
    ((_phase(field_raw_nopad), _phase(field_raw_pad)),
     "phase (rad)", "viridis"),
    ((_phase(field_spatial_nopad), _phase(field_spatial_pad)),
     "phase (rad)", "viridis"),
    ((_phase(field_fourier_nopad), _phase(field_fourier_pad)),
     "phase (rad)", "viridis"),
    ((diff_nopad, diff_pad),
     "Δphase (rad)", "RdBu_r"),
]
col_headers = ["padding=False (square crop)", "padding=True  (full hologram)"]

fig, axes = plt.subplots(4, 2, figsize=(8, 11), constrained_layout=True)
fig.suptitle("Spatial and Direct (Fourier) pipeline with and without padding\n"
             "No added tilt correction", fontsize=9)

for row_idx, ((img0, img1), cbar_label, cmap) in enumerate(rows):
    for col_idx, img in enumerate((img0, img1)):
        ax = axes[row_idx, col_idx]
        if cmap == "RdBu_r":
            # per-column limits: nopad column stays at its own (near-zero)
            # range so it appears uniformly dark; padded column shows full
            # contrast. annotate both with max |Δ| for direct comparison.
            lim = np.abs(img).max()
            im = ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim)
            ax.text(0.97, 0.03, f"max |Δ| = {lim:.2g} rad",
                    transform=ax.transAxes, fontsize=6.5, color="black",
                    ha="right", va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1.5))
        else:
            im = ax.imshow(img, cmap=cmap)
        ax.set_title(panel_titles[row_idx][col_idx], fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("x (px)", fontsize=7)
        fig.colorbar(im, ax=ax, label=cbar_label, shrink=0.85)
    axes[row_idx, 0].set_ylabel("y (px)", fontsize=7)

# column headers above row 0
for col_idx, header in enumerate(col_headers):
    axes[0, col_idx].annotate(
        header,
        xy=(0.5, 1.0), xycoords="axes fraction",
        xytext=(0, 28), textcoords="offset points",
        ha="center", va="bottom", fontsize=9, fontweight="bold",
    )

plt.savefig("pipeline_domain_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
