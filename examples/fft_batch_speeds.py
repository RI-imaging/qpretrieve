"""Fourier Transform speed benchmarks for OAH

This example visualizes the speed for different batch sizes for
the available FFT Filters. The y-axis shows the average speed of a pipeline
run for the Off-Axis Hologram class :class:`.OffAxisHologram`, including
background data processing. Therefore, four FFTs are run per pipeline.

- Optimum batch size is between 64 and 256 for 256x256pix imgs (incl padding),
  but will be limited by your computer's RAM.
- Here, batch size is the size of the 3D stack in z.
- Note that each pipeline runs 4 FFTs. For example, batch 8 runs 8*4=32 FFTs.

"""
import time
import matplotlib.pylab as plt
import numpy as np
import qpretrieve
from qpretrieve.data_array_layout import convert_data_to_3d_array_layout
from qpretrieve.fourier import FFTFilterNumpy, FFTFilterPyFFTW

# load the experimental data
edata = np.load("./data/hologram_cell.npz")

n_transforms_list = [8, 16, 32, 64, 128, 256]
subtract_mean = True
padding = True
# we take the PyFFTW speeds from the second run
fft_interfaces = [FFTFilterNumpy, FFTFilterPyFFTW, FFTFilterPyFFTW]
filter_name = "disk"
filter_size = 1 / 2
speed_norms = {}

# load and prep the data
data_2d = edata["data"].copy()
data_2d_bg = edata["bg_data"].copy()
data_3d_prep, _ = convert_data_to_3d_array_layout(data_2d)
data_3d_bg_prep, _ = convert_data_to_3d_array_layout(data_2d_bg)

for fft_interface in fft_interfaces:
    results = {}
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
        holo.run_pipeline(filter_name=filter_name, filter_size=filter_size)
        bg = qpretrieve.OffAxisHologram(data=data_3d_bg)
        bg.process_like(holo)
        t1 = time.time()
        results[n_transforms] = t1 - t0

    speed_norm = [timing / b_size for b_size, timing in results.items()]
    # the initial PyFFTW run (incl wisdom calc is overwritten here)
    speed_norms[fft_interface.__name__] = speed_norm

# setup plot
fig, axes = plt.subplots(1, 1, figsize=(8, 5))
ax1 = axes
width = 0.25  # the width of the bars
multiplier = 0
x_pos = np.arange(len(n_transforms_list))
colors = ["darkmagenta", "lightseagreen"]

for (name, speed), color in zip(speed_norms.items(), colors):
    offset = width * multiplier
    ax1.bar(x_pos + offset, speed, width, label=name,
            color=color, edgecolor='k')
    multiplier += 1

ax1.set_xticks(x_pos + (width / 2), labels=n_transforms_list)
ax1.set_xlabel("Input hologram batch size")
ax1.set_ylabel("OAH processing time [Time / batch size] (s)")
ax1.legend(loc='upper right', fontsize="large")

plt.suptitle("Batch processing time for Off-Axis Hologram\n"
             "(data+bg_data)")
plt.tight_layout()
# plt.show()
plt.savefig("fft_batch_speeds.png", dpi=150)
