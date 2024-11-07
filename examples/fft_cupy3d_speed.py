"""Fourier Transform speeds for the Cupy 3D interface

This example visualizes the normalised speed for different 3D image stacks for
the `FFTFilterCupy3D` FFT Filter

- Optimum stack size for 256 images (incl. padding) is bewteen 128 and 256.

"""
import time
import matplotlib.pylab as plt
import numpy as np
import qpretrieve

# load the experimental data
edata = np.load("./data/hologram_cell.npz")

n_transforms_list = [8, 16, 32, 64, 128, 256, 512]
subtract_mean = True
padding = True
fft_interface = qpretrieve.fourier.FFTFilterCupy3D

results = {}
for n_transforms in n_transforms_list:
    print(f"Running {n_transforms} transforms...")

    data_2d = edata["data"].copy()
    data_2d_bg = edata["bg_data"].copy()
    data_3d = np.repeat(
        edata["data"].copy()[np.newaxis, ...],
        repeats=n_transforms, axis=0)
    data_3d_bg = np.repeat(
        edata["bg_data"].copy()[np.newaxis, ...],
        repeats=n_transforms, axis=0)
    assert data_3d.shape == data_3d_bg.shape == (n_transforms,
                                                 edata["data"].shape[0],
                                                 edata["data"].shape[1])

    t0 = time.time()

    holo = qpretrieve.OffAxisHologram(data=data_3d,
                                      fft_interface=fft_interface,
                                      subtract_mean=subtract_mean, padding=padding)
    holo.run_pipeline(filter_name="disk", filter_size=1 / 2)
    bg = qpretrieve.OffAxisHologram(data=data_3d_bg)
    bg.process_like(holo)

    t1 = time.time()
    results[n_transforms] = t1 - t0

speed_norm = [v / k for k, v in results.items()]

fig, axes = plt.subplots(1, 1, figsize=(8, 5))
ax1 = axes

ax1.bar(range(len(n_transforms_list)), height=speed_norm, color='darkmagenta')
ax1.set_xticks(range(len(n_transforms_list)), labels=n_transforms_list)
ax1.set_xlabel("Number of Transforms")
ax1.set_ylabel("Speed normalised by number of transforms (s)")
ax1.set_title(f"Normalised by number of transforms")

plt.suptitle("Speed of CuPy 3D")
plt.tight_layout()
# plt.show()
plt.savefig("fft_cupy3d_speed.png", dpi=150)
