Setting the NDArray Backend
===========================

.. _sec_doc_ndarray_backend:

Since version 0.5.0, `qpretrieve` allows the user to leverage their CUDA GPU
via the :class:`.FFTFilterCupy` fft interface and the ``CuPy`` library.
However, the data up/download from/to the GPU was inefficient.
This was due to the use of CPU-bound numpy
arrays between GPU-based Fourier Transforms.

Since version 0.6.0, you can control the desired ndarray backend.
An "ndarray backend" is defined as the library used to define the
ndarrays in qpretrieve during runtime. By default it is set to ``'numpy'``.

If you are using the :class:`.FFTFilterCupy` fft interface, it is
recommended to set the backend to ``'cupy'``. See the script below and
the benchmarking example
``fft_batch_speeds.py`` in the :ref:`sec_examples` folder
for details on how to do this.
For more info, see the `CuPy library <https://cupy.dev/>`_.

There are currently two available ndarray backends: ``'numpy'`` and ``'cupy'``.

Controlling the ndarray backend
-------------------------------

``qpretrieve`` allows users to swap between these backends with the
:func:`qpretrieve.set_ndarray_backend()` function. To check which backend is
currently in use just run :func:`qpretrieve.get_ndarray_backend()`.

.. admonition:: Matching the NDArray Backend with the FFTFilter

	Always try to match the NDArray Backend with the FFTFilter, as shown in
	the example below, otherwise you will run into warnings or errors.

	To summarise:
		- 	``'numpy'`` (default) backend works as expected with the
			:class:`.FFTFilterNumpy` and :class:`.FFTFilterPyFFTW` classes.
		- 	``'cupy'`` backend works as expected with the :class:`.FFTFilterCupy`
			and :class:`.FFTFilterNumpy` classes. This is because NumPy is
			`quite clever <https://numpy.org/doc/stable/user/basics.interoperability.html#example-cupy-arrays>`_.



.. code-block:: python

	import qpretrieve

	print(qpretrieve.get_ndarray_backend())
	# <module 'numpy' from '~\\numpy\\__init__.py'>

	qpretrieve.set_ndarray_backend('cupy')  # swap to the 'cupy' backend
	print(qpretrieve.get_ndarray_backend())
	# <module 'cupy' from '~\\cupy\\__init__.py'>


Example use of 'cupy' backend for Off-Axis Hologram
---------------------------------------------------

.. code-block:: python

	import numpy as np
	import qpretrieve
	from qpretrieve.data_array_layout import convert_data_to_3d_array_layout

	# load your experimental data (data from the qpretrieve repository)
	edata = np.load("qpretrieve/examples/data/hologram_cell.npz")
	data_2d, data_2d_bg = edata["data"].copy(), edata["bg_data"].copy()
	input_data_3d, _ = convert_data_to_3d_array_layout(data_2d)
	input_data_bg_3d, _ = convert_data_to_3d_array_layout(data_2d_bg)

	# stack in 3D
	data_3d = np.repeat(input_data_3d, repeats=4, axis=0)
	data_3d_bg = np.repeat(input_data_bg_3d, repeats=4, axis=0)

	# set 'cupy' backend and fft_interface
	qpretrieve.set_ndarray_backend("cupy")
	fft_interface = qpretrieve.fourier.FFTFilterCupy

	# process the 3D stack of holograms
	holo = qpretrieve.OffAxisHologram(
		data_3d, fft_interface=fft_interface, padding=True)
	kwargs = dict(filter_name="disk", filter_size=1 / 3)
	holo_field = holo.run_pipeline(**kwargs)

	# process the 3D stack of background holograms
	bg = qpretrieve.OffAxisHologram(data_3d_bg, fft_interface=fft_interface)
	bg.process_like(holo)
