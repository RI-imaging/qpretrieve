"""Compare background correction before and after propagation.

Run from any working directory with::

    python examples/compare_spatial_and_propagated_background.py

The comparison plot is saved beside this script.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import nrefocus
import numpy as np
from skimage.restoration import unwrap_phase

import qpretrieve


EXAMPLE_DIR = Path(__file__).resolve().parent

# Set any color limit to a (vmin, vmax) tuple; None selects an automatic range.
PHASE_CLIM = None
AMPLITUDE_CLIM = None
PHASE_DIFFERENCE_CLIM = None
AMPLITUDE_DIFFERENCE_CLIM = None

# Square object ROI, centered in the image.
OBJECT_ROI_SIZE = 150


def main():
    data = np.load(EXAMPLE_DIR / "data" / "hologram_cell.npz")
    fft_interface = qpretrieve.fourier.FFTFilterNumpy

    sample_holo = qpretrieve.OffAxisHologram(
        data["data"], fft_interface=fft_interface
    )
    sample_fourier = sample_holo.run_pipeline(
        filter_name="smooth disk",
        filter_size=1 / 2,
        output_domain="fourier",
    )
    background_holo = qpretrieve.OffAxisHologram(
        data["bg_data"], fft_interface=fft_interface
    )
    background_holo.process_like(sample_holo)
    background_fourier = background_holo._fourier_field_data

    # Correct at the reconstruction plane, then propagate the corrected field.
    sample_field = sample_fourier.finalize()
    background_field = background_fourier.finalize()
    spatial_corrected_field = sample_field / background_field

    # Compare propagating the corrected field with propagating both fields
    # separately and correcting only after they reach the target plane.
    propagation_kwargs = dict(
        d=6, nm=1.533, res=8.25, method="fresnel", padding=False
    )
    # Use the same padded FFT grid as the Fourier-domain path. qpretrieve's
    # spatial fields are cropped back to the input image, so zero-pad the
    # corrected image to the original FFT shape before propagating it.
    height, width = spatial_corrected_field.shape[-2:]
    padded_corrected_field = np.zeros(
        (spatial_corrected_field.shape[0], *sample_fourier.fft_shape),
        dtype=spatial_corrected_field.dtype,
    )
    padded_corrected_field[:, :height, :width] = spatial_corrected_field
    corrected_propagated = nrefocus.refocus(
        field=padded_corrected_field,
        output_domain="spatial",
        **propagation_kwargs,
    )[:, :height, :width]
    sample_propagated = nrefocus.refocus(
        field=sample_fourier, output_domain="spatial", **propagation_kwargs
    )
    background_propagated = nrefocus.refocus(
        field=background_fourier,
        output_domain="spatial",
        **propagation_kwargs,
    )
    propagated_ratio = sample_propagated / background_propagated
    phase_correct_then_propagate = unwrap_phase(
        np.angle(corrected_propagated[0])
    )
    amplitude_correct_then_propagate = np.abs(corrected_propagated[0])
    phase_propagate_then_correct = unwrap_phase(np.angle(propagated_ratio[0]))
    amplitude_propagate_then_correct = np.abs(propagated_ratio[0])

    # The two unwrapping runs can differ by a global multiple of 2*pi.
    phase_difference = (
        phase_propagate_then_correct - phase_correct_then_propagate
    )
    phase_difference -= 2 * np.pi * np.round(
        np.median(phase_difference) / (2 * np.pi)
    )
    amplitude_difference = np.abs(
        amplitude_propagate_then_correct - amplitude_correct_then_propagate
    )

    phase_images = [
        phase_correct_then_propagate,
        phase_propagate_then_correct,
    ]
    amplitude_images = [
        amplitude_correct_then_propagate,
        amplitude_propagate_then_correct,
    ]
    phase_auto = max(np.percentile(np.abs(p), 99) for p in phase_images)
    phase_clim = PHASE_CLIM or (-phase_auto, phase_auto)
    amplitude_auto = (
        min(np.percentile(a, 1) for a in amplitude_images),
        max(np.percentile(a, 99) for a in amplitude_images),
    )
    amplitude_clim = AMPLITUDE_CLIM or amplitude_auto
    phase_difference_auto = np.percentile(np.abs(phase_difference), 99)
    phase_difference_clim = PHASE_DIFFERENCE_CLIM or (
        -phase_difference_auto, phase_difference_auto
    )
    amplitude_difference_auto = np.percentile(
        np.abs(amplitude_difference), 99
    )
    amplitude_difference_clim = AMPLITUDE_DIFFERENCE_CLIM or (
        0, amplitude_difference_auto
    )

    height, width = phase_difference.shape
    roi_x = (width - OBJECT_ROI_SIZE) // 2
    roi_y = (height - OBJECT_ROI_SIZE) // 2
    roi = np.zeros((height, width), dtype=bool)
    roi[
        roi_y:roi_y + OBJECT_ROI_SIZE,
        roi_x:roi_x + OBJECT_ROI_SIZE,
    ] = True

    def roi_stats(image):
        values = image[roi]
        return float(np.mean(values)), float(np.std(values))

    phase_a_stats = roi_stats(phase_images[0])
    phase_b_stats = roi_stats(phase_images[1])
    phase_delta_stats = roi_stats(phase_difference)
    amplitude_a_stats = roi_stats(amplitude_images[0])
    amplitude_b_stats = roi_stats(amplitude_images[1])
    amplitude_delta_stats = roi_stats(amplitude_difference)

    table_rows = [
        ["Phase mean [rad]", f"{phase_a_stats[0]:.4f}",
         f"{phase_b_stats[0]:.4f}",
         f"{phase_b_stats[0] - phase_a_stats[0]:.4f}"],
        ["Phase SD [rad]", f"{phase_a_stats[1]:.4f}",
         f"{phase_b_stats[1]:.4f}",
         f"{phase_b_stats[1] - phase_a_stats[1]:.4f}"],
        ["Amplitude mean", f"{amplitude_a_stats[0]:.4f}",
         f"{amplitude_b_stats[0]:.4f}",
         f"{amplitude_b_stats[0] - amplitude_a_stats[0]:.4f}"],
        ["Amplitude SD", f"{amplitude_a_stats[1]:.4f}",
         f"{amplitude_b_stats[1]:.4f}",
         f"{amplitude_b_stats[1] - amplitude_a_stats[1]:.4f}"],
        ["Phase difference map mean +/- SD [rad]", "-", "-",
         f"{phase_delta_stats[0]:.4f} +/- {phase_delta_stats[1]:.4f}"],
        ["Absolute amplitude difference mean +/- SD", "-", "-",
         f"{amplitude_delta_stats[0]:.4f} +/- "
         f"{amplitude_delta_stats[1]:.4f}"],
    ]
    table_headers = [
        "Statistic",
        "Correct, then propagate",
        "Propagate, then correct",
        "Change / difference map",
    ]
    column_widths = [
        max(len(header), *(len(row[index]) for row in table_rows))
        for index, header in enumerate(table_headers)
    ]

    def print_table_border():
        print("+" + "+".join("-" * (width + 2) for width in column_widths)
              + "+")

    def print_table_row(row):
        print("|" + "|".join(
            f" {value:<{width}} "
            for value, width in zip(row, column_widths)
        ) + "|")

    print_table_border()
    print_table_row(table_headers)
    print_table_border()
    for row in table_rows:
        print_table_row(row)
    print_table_border()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14, 7),
        constrained_layout=True,
    )
    titles = [
        "Correct, then propagate",
        "Propagate, then correct",
        "Difference",
    ]
    phase_panels = [
        phase_correct_then_propagate,
        phase_propagate_then_correct,
        phase_difference,
    ]
    amplitude_panels = [
        amplitude_correct_then_propagate,
        amplitude_propagate_then_correct,
        amplitude_difference,
    ]

    for col, title in enumerate(titles):
        phase_range = phase_clim if col < 2 else phase_difference_clim
        phase_plot = axes[0, col].imshow(
            phase_panels[col],
            cmap="coolwarm" if col < 2 else "RdBu_r",
            vmin=phase_range[0],
            vmax=phase_range[1],
        )
        axes[0, col].set_title(title)
        fig.colorbar(
            phase_plot,
            ax=axes[0, col],
            label=(
                "difference of unwrapped phases [rad]"
                if col == 2 else "phase [rad] (unwrapped)"
            ),
        )

        amp_range = (
            amplitude_clim if col < 2 else amplitude_difference_clim
        )
        amp_plot = axes[1, col].imshow(
            amplitude_panels[col],
            cmap="viridis",
            vmin=amp_range[0],
            vmax=amp_range[1],
        )
        fig.colorbar(
            amp_plot,
            ax=axes[1, col],
            label=(
                "absolute amplitude difference"
                if col == 2 else "amplitude ratio"
            ),
        )

        if col < 2:
            for row in (0, 1):
                axes[row, col].add_patch(
                    Rectangle(
                        (roi_x, roi_y),
                        width=OBJECT_ROI_SIZE,
                        height=OBJECT_ROI_SIZE,
                        fill=False,
                        edgecolor="white",
                        linewidth=1.2,
                    )
                )

    axes[0, 0].set_ylabel("Phase")
    axes[1, 0].set_ylabel("Amplitude")
    for ax in axes.flat:
        ax.set_axis_off()

    fig.suptitle(
        "Correction order compared at the same propagated plane",
        fontsize=14,
    )
    fig.savefig(
        EXAMPLE_DIR / "compare_spatial_and_propagated_background.png", dpi=180
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
