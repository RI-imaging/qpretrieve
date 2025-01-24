Data Array Layouts
==================

.. _sec_doc_array_layout:

Since version 0.4.0, `qpretrieve` accepts 3D (z,y,x) arrays as input.
Additionally, it **always** returns in data as the 3D array layout.

We use the term "*array layout*" to define the different ways to represent data.
The currently accepted array layouts are: 2D, RGB, RGBA, 3D.

Summary of allowed Array Layouts::

    Input         ->  Output
    2D   (y,x)    ->  3D (1,y,x)
    RGB  (y,x,3)  ->  3D (1,y,x)
    RGBA (y,x,4)  ->  3D (1,y,x)
    3D   (z,y,x)  ->  3D (z,y,x)

Notes on RGB/RGBA Array Layouts
-------------------------------

If you give either a RGB or RGBA array layout as input, then the first
channel is taken as the image to process. In other words, it is assumed that
all channels contain the same information, so the first channel is used.

If you use the `oah.get_array_with_input_layout("phase")` method for
the RGBA array layout, then the alpha (A) channel will be an array of ones.

Converting to and from Array Layouts
------------------------------------

`qpretrieve` will automatically handle the above allowed array layouts.
In other words, if you provide any 2D, RGB, RGBA, or 3D data as input to
:class:`.OffAxisHologram` or :class:`.QLSInterferogram`
the class will handle everything for you.

However, if you want to have your processed data in the same array layout as when
you inputted it, then you can use the convenience method
:meth:`get_array_with_input_layout` to do just that. For example, if
your input data was a 2D array, you can get the processed field, phase,
amplitude etc like so:

.. code-block:: python

	# 2D data inputted
	oah = qpretrieve.OffAxisHologram(hologram_2d)
	# do some processing
	...
	# get your data as a 2D array layout
	field_2d = oah.get_array_with_input_layout("field")
	phase_2d = oah.get_array_with_input_layout("phase")
	amplitude_2d = oah.get_array_with_input_layout("amplitude")