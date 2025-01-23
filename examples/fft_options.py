"""Available Fourier Transform backends (`FFTFilter`)

This example visualizes the different backends and packages available to the
user for performing Fourier transforms.

- PyFFTW is initially slow, but over many FFTs is very quick.

"""
import time
import matplotlib.pylab as plt
import numpy as np
import qpretrieve
from qpretrieve.data_array_layout import convert_data_to_3d_array_layout

# load the experimental data
edata = np.load("./data/hologram_cell.npz")

# get the available fft interfaces
interfaces_available = qpretrieve.fourier.get_available_interfaces()
num_interfaces = len(interfaces_available)
fft_interfaces = [interf.__name__ for interf in interfaces_available]

n_transforms = 200
subtract_mean = True
padding = True
filter_name = "disk"
filter_size = 1 / 2

# load and prep the data
data_2d = edata["data"].copy()
data_2d_bg = edata["bg_data"].copy()
data_3d_prep, _ = convert_data_to_3d_array_layout(data_2d)
data_3d_bg_prep, _ = convert_data_to_3d_array_layout(data_2d_bg)

print("Running single transform...")
# one transform
results_1 = {}
for fft_interface in interfaces_available:
    t0 = time.time()
    holo = qpretrieve.OffAxisHologram(data=data_2d,
                                      fft_interface=fft_interface,
                                      subtract_mean=subtract_mean,
                                      padding=padding)
    holo.run_pipeline(filter_name=filter_name, filter_size=filter_size)
    bg = qpretrieve.OffAxisHologram(data=data_2d_bg)
    bg.process_like(holo)
    t1 = time.time()
    results_1[fft_interface.__name__] = t1 - t0
num_interfaces = len(results_1)

# multiple transforms repeated in 2D
print(f"Running {n_transforms} transforms in a loop...")
results_2d = {}
for fft_interface in interfaces_available:

    t0 = time.time()
    for _ in range(n_transforms):
        assert data_3d_prep.shape == data_3d_bg_prep.shape == (
            1, edata["data"].shape[0], edata["data"].shape[1])

        holo = qpretrieve.OffAxisHologram(data=data_3d_prep,
                                          fft_interface=fft_interface,
                                          subtract_mean=subtract_mean,
                                          padding=padding)
        holo.run_pipeline(filter_name=filter_name, filter_size=filter_size)
        bg = qpretrieve.OffAxisHologram(data=data_3d_bg_prep)
        bg.process_like(holo)
    t1 = time.time()
    results_2d[fft_interface.__name__] = t1 - t0

# multiple transforms in 3D
print(f"Running {n_transforms} transforms at once...")
results_3d = {}
for fft_interface in interfaces_available:
    # create batch
    data_3d = np.repeat(data_3d_prep, repeats=n_transforms, axis=0)
    data_3d_bg = np.repeat(data_3d_bg_prep, repeats=n_transforms, axis=0)

    assert data_3d.shape == data_3d_bg.shape == (
        n_transforms, edata["data"].shape[0], edata["data"].shape[1])

    t0 = time.time()
    holo = qpretrieve.OffAxisHologram(data=data_3d,
                                      fft_interface=fft_interface,
                                      subtract_mean=subtract_mean,
                                      padding=padding)
    holo.run_pipeline(filter_name=filter_name, filter_size=filter_size)
    bg = qpretrieve.OffAxisHologram(data=data_3d_bg)
    bg.process_like(holo)
    t1 = time.time()
    results_3d[fft_interface.__name__] = t1 - t0

speed_1 = list(results_1.values())
speed_2d = list(results_2d.values())
speed_3d = list(results_3d.values())

fig, axes = plt.subplots(1, 3, figsize=(12, 5))
ax1, ax2, ax3 = axes
labels = [fftstr[9:] for fftstr in fft_interfaces]
colors = ["lightseagreen", "darkmagenta"]

ax1.bar(range(num_interfaces), height=speed_1, color=colors, edgecolor='k')
ax1.set_xticks(range(num_interfaces), labels=labels)
ax1.set_ylabel("Time (s)")
ax1.set_title("1 Transform")

ax2.bar(range(num_interfaces), height=speed_2d, color=colors, edgecolor='k')
ax2.set_xticks(range(num_interfaces), labels=labels)
ax2.set_ylabel("Time (s)")
ax2.set_title(f"{n_transforms} Transforms (one-by-one)")
ax2.set_ylim(0, 4.5)

ax3.bar(range(num_interfaces), height=speed_3d, color=colors, edgecolor='k')
ax3.set_xticks(range(num_interfaces), labels=labels)
ax3.set_ylabel("Time (s)")
ax3.set_title(f"{n_transforms} Transforms (single 3D batch)")
ax3.set_ylim(0, 4.5)

plt.suptitle("Processing time for FFT Filters")
plt.tight_layout()
# plt.show()
plt.savefig("fft_options.png", dpi=150)
