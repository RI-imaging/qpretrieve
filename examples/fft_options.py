"""
Fourier Transform interfaces available.

This example visualizes the different backends and packages available to the
user for performing Fourier transforms.

Notes

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

# multiple transforms (should see speed increase for PyFFTW)
print(f"Running {n_transforms} transforms...")
results = {}
for fft_interface in interfaces_available:

    # create batch
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

    results[fft_interface.__name__] = t1 - t0

num_interfaces = len(results)
fft_interfaces = list(results.keys())
speed_1 = list(results_1.values())
speed = list(results.values())

fig, axes = plt.subplots(1, 2, figsize=(8, 5))
ax1, ax2 = axes
labels = [fftstr[9:] for fftstr in fft_interfaces]
colors = ["lightseagreen", "darkmagenta"]

ax1.bar(range(num_interfaces), height=speed_1, color=colors, edgecolor='k')
ax1.set_xticks(range(num_interfaces), labels=labels)
ax1.set_ylabel("Time (s)")
ax1.set_title("1 Transform")

ax2.bar(range(num_interfaces), height=speed, color=colors, edgecolor='k')
ax2.set_xticks(range(num_interfaces), labels=labels,
               rotation=45)
ax2.set_ylabel("Time (s)")
# todo: fix code, then this title
ax2.set_title(f"{n_transforms} Transforms (single 3D batch)")

plt.suptitle("Processing time for FFT Filters")
plt.tight_layout()
# plt.show()
plt.savefig("fft_options.png", dpi=150)
