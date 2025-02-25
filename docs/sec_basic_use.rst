Basic Use of qpretrieve
=======================

To run this code locally, you can download the example dataset from the
`qpretrieve repository <https://github.com/RI-imaging/qpretrieve/blob/main/examples/data/hologram_cell.npz>`_


.. code-block:: python

	import qpretrieve

	# load your experimental data (data from the qpretrieve repository)
	edata = np.load("qpretrieve/examples/data/hologram_cell.npz")
	data = edata["data"]
	data_bg = edata["bg_data"]

	# create an off-axis hologram object, to process the holography data
	oah = qpretrieve.OffAxisHologram(data)
	oah.run_pipeline(filter_name=filter_name, filter_size=filter_size)
	# process background hologram
	oah_bg = qpretrieve.OffAxisHologram(data=data_2d_bg)
	oah_bg.process_like(oah)

	print(f"Original Hologram shape: {data.shape}")
	print(f"Processed Field shape: {oah.field.shape}")

	# Now you can look at the phase data
	phase_corrected = oah.phase - oah_bg.phase



How qpretrieve now handles data
-------------------------------

In older versions of qpretrieve, only a single image could be processed at a time.
Since version (0.4.0), it accepts 3D array layouts, and always returns
the data as 3D, regardless of the input array layout.

Click here for more :ref:`information on array layouts <sec_doc_array_layout>`.
There are speed benchmarks given for different inputs in the
:ref:`examples <sec_examples>` section.


New version example code and output:
....................................

.. code-block:: python

	import numpy as np
	import qpretrieve

	hologram = np.ones(shape=(256, 256))
	oah = qpretrieve.OffAxisHologram(hologram)
	oah.run_pipeline()
	assert oah.field.shape == (1, 256, 256)  # <- now a 3D array is returned
	# if you want the original array layout (2d)
	field_2d = oah.get_data_with_input_layout("field")
	assert field_2d.shape == (256, 256)

	# this means you can input 3D arrays
	hologram_3d = np.ones(shape=(50, 256, 256))
	oah = qpretrieve.OffAxisHologram(hologram_3d)
	oah.run_pipeline()
	assert oah.field.shape == (50, 256, 256)  # <- always a 3D array


Old version example code and output:
....................................

.. code-block:: python

	import numpy as np
	import qpretrieve  # versions older than 0.4.0

	hologram = np.ones(shape=(256, 256))
	oah = qpretrieve.OffAxisHologram(hologram)
	oah.run_pipeline()
	assert oah.field.shape == hologram.shape  # <- old version returned a 2D array
