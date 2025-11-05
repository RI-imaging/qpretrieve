"""Fourier Transform speed benchmarks for OAH

.. admonition:: Benchmark your own data

    You can use this script to benchmark your data and find the optimum batch
    size and FFTFilter. Just load your data in place of the ``input_data_3d``
    and ``input_data_bg_3d``.


This example visualizes the speed for different batch sizes for
the available FFT Filters.

In the first graph "Batch Processing Time",
the y-axis shows the speed - normalised by batch size -
of an Off-Axis Hologram class :class:`.OffAxisHologram`,
including background data processing. From this graph, we can conclude that:

   - Optimum batch size is between 32 and 256 for 256x256pix imgs
     (incl padding), but will be limited by your computer's RAM.
   - Some Notes:
      - The batch size is the size of the raw hologram 3D stack in z.
      - Each pipeline runs 4 FFTs (Data FFT and iFFT + background Data FFT +
        iFFT). For example, batch size of 8 runs 8*4=32 FFTs.
      - For CuPy, the data transfer between GPU and CPU is
        currently very inefficient.


In the second graph "FFT Speed for Off-Axis Hologram",
the y-axis shows the speed of a single FFT.
From this graph, we can conclude that:

   - Optimum batch size for CuPy and PyFFTW is between 16 and 64 for
     256x256pix imgs (incl padding). This may change depending on your machine.


.. admonition:: Notes on why each FFT method is run twice

    In the first run, python must load extra modules, which makes
    benchmarking unreliable. Therefore, the second run is used for timing.

    This is especially important for PyFFTW, as PyFFTW generates a "wisdom"
    on its first run, which defines the fastest way to do subsequent
    Fourier transforms. A wisdom is generated for each batch size.
    Click here for more information on
    `exporting and importing wisdoms with PyFFTW
    <https://pyfftw.readthedocs.io/en/latest/source/pyfftw/
    pyfftw.html#pyfftw.export_wisdom>`_.

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

# the first run is always used as a warmup
fft_interfaces = [FFTFilterNumpy, FFTFilterNumpy]
if FFTFilterPyFFTW is not None:
    fft_interfaces.extend([FFTFilterPyFFTW, FFTFilterPyFFTW])
if FFTFilterCupy is not None:
    fft_interfaces.extend([FFTFilterCupy, FFTFilterCupy])

filter_name = "disk"
filter_size = 1 / 2

# load and prep the data
data_2d, data_2d_bg = edata["data"].copy(), edata["bg_data"].copy()
input_data_3d, _ = convert_data_to_3d_array_layout(data_2d)
input_data_bg_3d, _ = convert_data_to_3d_array_layout(data_2d_bg)

speed_batch_norms, speed_ffts = {}, {}
print("Each FFTFilter will run twice, as a warmup run is "
      "required for comparison.")
for fft_interface in fft_interfaces:
    results_batch, results_fft = {}, {}
    for n_transforms in n_transforms_list:
        print(f"Running {n_transforms} transforms with "
              f"{fft_interface.__name__}")

        # create batches
        data_3d = np.repeat(input_data_3d, repeats=n_transforms, axis=0)
        data_3d_bg = np.repeat(input_data_bg_3d, repeats=n_transforms, axis=0)

        assert data_3d.shape == data_3d_bg.shape == (
            n_transforms, edata["data"].shape[0], edata["data"].shape[1])

        t0 = time.perf_counter()
        holo = qpretrieve.OffAxisHologram(data=data_3d,
                                          fft_interface=fft_interface,
                                          subtract_mean=subtract_mean,
                                          padding=padding)
        t_fft = time.perf_counter()
        holo.run_pipeline(filter_name=filter_name, filter_size=filter_size)
        bg = qpretrieve.OffAxisHologram(data=data_3d_bg)
        bg.process_like(holo)
        t_batch = time.perf_counter()
        results_batch[n_transforms] = t_batch - t0
        results_fft[n_transforms] = t_fft - t0

    speed_batch_norm = [t / bsize for bsize, t in results_batch.items()]
    speed_fft = [t for t in results_fft.values()]
    # the initial PyFFTW run (incl wisdom calc is overwritten here)
    speed_batch_norms[fft_interface.__name__] = speed_batch_norm
    speed_ffts[fft_interface.__name__] = speed_fft

# setup figure
width = 0.25  # the width of the bars
x_pos = np.arange(len(n_transforms_list))
colors = ["darkmagenta", "lightseagreen", "goldenrod"]
edgecolor = "k"
legend_loc = "upper center"
fontsize = 16

fig, axes = plt.subplots(2, 1, figsize=(8, 10))
ax1, ax2 = axes

# setup plot for batch speed comparison
multiplier = 0  # for the x_label positions
for (name, speed), color in zip(speed_batch_norms.items(), colors):
    offset = width * multiplier
    ax1.bar(x_pos + offset, speed, width, label=name,
            color=color, edgecolor=edgecolor)
    multiplier += 1
ax1.set_xticks(x_pos + width, labels=n_transforms_list)
ax1.set_xlabel("Input hologram batch size")
ax1.set_ylabel("OAH processing time [Time / batch size] (s)")
ax1.legend(loc=legend_loc, fontsize="large")
ax1.set_title("Batch Process Time for Off-Axis Hologram "
              "(data+bg_data)\n(after PyFFTW warmup)", fontsize=fontsize)

# setup plot for fft speed comparison
multiplier = 0  # for the x_label positions
for (name, speed), color in zip(speed_ffts.items(), colors):
    offset = width * multiplier
    ax2.bar(x_pos + offset, speed, width, label=name,
            color=color, edgecolor=edgecolor)
    multiplier += 1
ax2.set_xticks(x_pos + width, labels=n_transforms_list)
ax2.set_xlabel("Input hologram batch size")
ax2.set_ylabel("FFT processing time (s)")
ax2.legend(loc=legend_loc, fontsize="large")
ax2.set_title("FFT Speed for Off-Axis Hologram\n(after PyFFTW warmup)",
              fontsize=fontsize)

plt.tight_layout()
# plt.show()
plt.savefig("fft_batch_speeds.png", dpi=150)
