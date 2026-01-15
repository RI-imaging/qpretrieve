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

   - Optimum batch size is between 32 and 128 for 256x256pix imgs
     (incl padding), but will be limited by your computer's RAM.
   - Some Notes:
      - The batch size is the size of the raw hologram 3D stack in z.
      - Each pipeline runs 4 FFTs (Data FFT and iFFT + background Data FFT and
        iFFT). For example, batch size of 8 runs 8*4=32 FFTs.
      - **For CuPy, the graphs do not include the time taken to transfer
        the GPU arrays back to the CPU.**


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

    ***large value artifact** for batch size 8 for FFTFilterCupy removed
    from graph for between comparison

.. admonition:: Notes on used CPU and GPU

    - Machine: LENOVO ThinkPad T15g Gen 1, Windows 11 Enterprise
    - CPU: Core i9-10885H, 32 GB RAM
    - GPU: NVIDIA GeForce RTX 2070 Super with Max-Q Design, 24 GB Memory

"""
import time
import matplotlib.pylab as plt
import numpy as np
import qpretrieve
from qpretrieve.data_array_layout import convert_data_to_3d_array_layout
from qpretrieve.fourier import FFTFilterNumpy, FFTFilterPyFFTW, FFTFilterCupy

# load the experimental data
edata = np.load("./data/hologram_cell.npz")

n_transforms_list = [8, 16, 32, 64, 128]
subtract_mean = True
padding = True

# the first run is always used as a warmup
fft_interfaces = [FFTFilterNumpy, FFTFilterNumpy]
if FFTFilterPyFFTW is not None:
    fft_interfaces.extend([FFTFilterPyFFTW, FFTFilterPyFFTW])
if FFTFilterCupy is not None:
    # the third run is for the CPU Download
    fft_interfaces.extend([FFTFilterCupy, FFTFilterCupy, FFTFilterCupy])

filter_name = "disk"
filter_size = 1 / 2

# load and prep the data
data_2d, data_2d_bg = edata["data"].copy(), edata["bg_data"].copy()
input_data_3d, _ = convert_data_to_3d_array_layout(data_2d)
input_data_bg_3d, _ = convert_data_to_3d_array_layout(data_2d_bg)

speed_batch_norms, speed_ffts = {}, {}
print("Each `FFTFilter` will run twice, as the first run often needs to do "
      "extra tasks, such as module loading or PyFFTW wisdom generation.")
for i, fft_interface in enumerate(fft_interfaces):
    fft_iface_name = fft_interface.__name__
    results_batch, results_fft = {}, {}

    if "FFTFilterCupy" in fft_iface_name:
        qpretrieve.set_ndarray_backend("cupy")
        if i == len(fft_interfaces) - 1:
            fft_iface_name = fft_iface_name + " + CPU Download"
    else:
        qpretrieve.set_ndarray_backend("numpy")

    for n_transforms in n_transforms_list:
        print(f"Running {n_transforms} transforms with "
              f"{fft_iface_name}")

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

        t_fft = time.perf_counter() - t0
        holo.run_pipeline(filter_name=filter_name, filter_size=filter_size)
        bg = qpretrieve.OffAxisHologram(data=data_3d_bg,
                                        fft_interface=fft_interface)
        bg.process_like(holo)
        t_batch = time.perf_counter() - t0

        t_dwnld_fft, t_dwnld_fld = 0, 0
        if "CPU Download" in fft_iface_name:
            t_dwnld_0 = time.perf_counter()
            # example for getting an FFT array onto the CPU
            fft_cpu = holo.fft_filtered.get()
            t_dwnld_fft = time.perf_counter() - t_dwnld_0
            # example for getting a Field array onto the CPU
            field_cpu = holo.field.get()
            t_dwnld_fld = time.perf_counter() - t_dwnld_0

        # artifact occurs for batch 8 with cupy
        if n_transforms == 8 and "FFTFilterCupy" in fft_iface_name:
            results_batch[n_transforms] = 0
            results_fft[n_transforms] = 0
        else:
            results_batch[n_transforms] = t_batch + t_dwnld_fld
            results_fft[n_transforms] = t_fft + t_dwnld_fft

    speed_batch_norm = [t / bsize for bsize, t in results_batch.items()]
    speed_fft = [t for t in results_fft.values()]
    # the initial PyFFTW run (incl wisdom calc is overwritten here)
    speed_batch_norms[fft_iface_name] = speed_batch_norm
    speed_ffts[fft_iface_name] = speed_fft

# setup figure
width = 0.2  # the width of the bars
x_pos = np.arange(len(n_transforms_list))
n_labels_list = [str(n) + "*" if n == 8 else str(n) for n in n_transforms_list]
colors = ["darkmagenta", "lightseagreen", "goldenrod", "goldenrod"]
hatches = ["", "", "", "//"]
edgecolor = "k"
legend_loc = "upper center"
fontsize = 16

fig, axes = plt.subplots(2, 1, figsize=(8, 10))
ax1, ax2 = axes

# setup plot for batch speed comparison
multiplier = -0.5  # for the x_label positions
for (name, speed), color, hatch in zip(speed_batch_norms.items(), colors, hatches):
    offset = width * multiplier
    ax1.bar(x_pos + offset, speed, width, label=name,
            color=color, edgecolor=edgecolor, hatch=hatch)
    multiplier += 1
ax1.set_xticks(x_pos + width, labels=n_labels_list)
ax1.set_xlabel("Input hologram batch size")
ax1.set_ylabel("OAH processing time [Time / batch size] (s)")
ax1.legend(loc=legend_loc, fontsize="large")
ax1.set_title("Batch Process Time for Off-Axis Hologram "
              "(data+bg_data)\n(after PyFFTW warmup)", fontsize=fontsize)

# setup plot for fft speed comparison
multiplier = -0.5  # for the x_label positions
for (name, speed), color, hatch in zip(speed_ffts.items(), colors, hatches):
    offset = width * multiplier
    ax2.bar(x_pos + offset, speed, width, label=name,
            color=color, edgecolor=edgecolor, hatch=hatch)
    multiplier += 1
ax2.set_xticks(x_pos + width, labels=n_labels_list)
ax2.set_xlabel("Input hologram batch size")
ax2.set_ylabel("FFT processing time (s)")
ax2.legend(loc=legend_loc, fontsize="large")
ax2.set_title("FFT Speed for Off-Axis Hologram\n(after PyFFTW warmup)",
              fontsize=fontsize)

plt.tight_layout()
plt.show()
# plt.savefig("fft_batch_speeds.png", dpi=150)
