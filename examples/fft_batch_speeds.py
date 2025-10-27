"""Fourier Transform speed benchmarks for OAH

This example visualizes the speed for different batch sizes for
the available FFT Filters. In both cases, a warmup run for the PyFFTW FFTFilter
is run before recording the speed. Always keep a single batch size when using PyFFTW.

In the first graph "Normalised Batch Processing Time",
the y-axis shows the speed - normalised by batch size -
of an Off-Axis Hologram class :class:`.OffAxisHologram`,
including background data processing. From this graph, we can conclude that:

   - Optimum batch size is between 32 and 256 for 256x256pix imgs (incl padding),
     but will be limited by your computer's RAM.
   - Here, batch size is the size of the raw hologram  3D stack in z.
   - Note that each pipeline runs 4 FFTs. For example, batch 8 runs 8*4=32 FFTs.
   - Also Note that for CuPy, the data transfer between GPU and CPU is
     currently very inefficient.

In the second graph "FFT Speed for Off-Axis Hologram",
the y-axis shows the speed of a single FFT.
From this graph, we can conclude that:

   - Optimum batch size for CuPy and PyFFTW is ~32 for 256x256pix imgs (incl padding).
     This may change depending on your machine.

"""
import time
import matplotlib.pylab as plt
import numpy as np
import qpretrieve
from qpretrieve.data_array_layout import convert_data_to_3d_array_layout
from qpretrieve.fourier import FFTFilterNumpy, FFTFilterPyFFTW, FFTFilterCupy

# load the experimental data
edata = np.load("./data/hologram_cell.npz")

n_transforms_list = [8, 16, 32, 64, 128, 256]
subtract_mean = True
padding = True
# we take the PyFFTW speeds from the second run
fft_interfaces = [FFTFilterNumpy, FFTFilterPyFFTW, FFTFilterPyFFTW,
                  FFTFilterCupy]
filter_name = "disk"
filter_size = 1 / 2
speed_batch_norms, speed_fft_norms = {}, {}

# load and prep the data
data_2d = edata["data"].copy()
data_2d_bg = edata["bg_data"].copy()
data_3d_prep, _ = convert_data_to_3d_array_layout(data_2d)
data_3d_bg_prep, _ = convert_data_to_3d_array_layout(data_2d_bg)

print("PyFFTW will run twice, as a warmup run is required for comparison.")

for fft_interface in fft_interfaces:
    results_batch, results_fft = {}, {}
    for n_transforms in n_transforms_list:
        print(f"Running {n_transforms} transforms with "
              f"{fft_interface.__name__}")

        # create batches
        data_3d = np.repeat(data_3d_prep, repeats=n_transforms, axis=0)
        data_3d_bg = np.repeat(data_3d_bg_prep, repeats=n_transforms, axis=0)

        assert data_3d.shape == data_3d_bg.shape == (n_transforms,
                                                     edata["data"].shape[0],
                                                     edata["data"].shape[1])

        t0 = time.time()
        holo = qpretrieve.OffAxisHologram(data=data_3d,
                                          fft_interface=fft_interface,
                                          subtract_mean=subtract_mean,
                                          padding=padding)
        t_fft = time.time()
        holo.run_pipeline(filter_name=filter_name, filter_size=filter_size)
        bg = qpretrieve.OffAxisHologram(data=data_3d_bg)
        bg.process_like(holo)
        t_batch = time.time()
        results_batch[n_transforms] = t_batch - t0
        results_fft[n_transforms] = t_fft - t0

    speed_batch_norm = [timing / b_size for b_size, timing in results_batch.items()]
    speed_fft_norm = [timing for timing in results_fft.values()]
    # the initial PyFFTW run (incl wisdom calc is overwritten here)
    speed_batch_norms[fft_interface.__name__] = speed_batch_norm
    speed_fft_norms[fft_interface.__name__] = speed_fft_norm

width = 0.25  # the width of the bars
x_pos = np.arange(len(n_transforms_list))
colors = ["darkmagenta", "lightseagreen", "goldenrod"]
legend_loc = 'upper center'

#### setup plot for batch speed comparison ####
multiplier = 0
fig, axes = plt.subplots(1, 1, figsize=(8, 5))
ax1 = axes

for (name, speed), color in zip(speed_batch_norms.items(), colors):
    offset = width * multiplier
    ax1.bar(x_pos + offset, speed, width, label=name,
            color=color, edgecolor='k')
    multiplier += 1
ax1.set_xticks(x_pos + width, labels=n_transforms_list)
ax1.set_xlabel("Input hologram batch size")
ax1.set_ylabel("OAH processing time [Time / batch size] (s)")
ax1.legend(loc=legend_loc, fontsize="large")
plt.suptitle("Normalised Batch Processing Time for Off-Axis Hologram "
             "(data+bg_data)\n(after PyFFTW warmup)")
plt.tight_layout()
plt.savefig("OAH_process_batch_speeds.png", dpi=150)

#### setup plot for fft speed comparison ####
multiplier = 0
fig, axes = plt.subplots(1, 1, figsize=(8, 5))
ax1 = axes

for (name, speed), color in zip(speed_fft_norms.items(), colors):
    offset = width * multiplier
    ax1.bar(x_pos + offset, speed, width, label=name,
            color=color, edgecolor='k')
    multiplier += 1
ax1.set_xticks(x_pos + width, labels=n_transforms_list)
ax1.set_xlabel("Input hologram batch size")
ax1.set_ylabel("FFT processing time (s)")
ax1.legend(loc=legend_loc, fontsize="large")
plt.suptitle("FFT Speed for Off-Axis Hologram\n(after PyFFTW warmup)")
plt.tight_layout()
plt.savefig("OAH_fft_speeds.png", dpi=150)
