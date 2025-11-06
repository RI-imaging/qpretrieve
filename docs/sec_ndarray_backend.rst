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
the benchmarking example ``fft_batch_speeds.py`` in the examples folder
for details on how to do this.
For more info, see the `CuPy library <https://cupy.dev/>`_.

There are currently two available ndarray backends: ``'numpy'`` and ``'cupy'``.

Controlling the ndarray backend
-------------------------------

``qpretrieve`` allows users to swap between these backends with the
:func:`qpretrieve.set_ndarray_backend()` function. To check which backend is
currently in use just run :func:`.qpretrieve.get_ndarray_backend()`.

.. code-block:: python

	import qpretrieve

	print(qpretrieve.get_ndarray_backend().__name__)
	# > 'numpy'

	qpretrieve.set_ndarray_backend('cupy')  # swap to the 'cupy' backend
	print(qpretrieve.get_ndarray_backend().__name__)
	# > 'cupy'
