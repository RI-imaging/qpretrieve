Data Array Layouts
==================

.. _sec_doc_array_layout:

Since version 0.4.0, `qpretrieve` accepts 3D (z,y,x) arrays as input.
Additionally, it **always** returns data as the 3D array layout.

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

**Inputting RGB(A)**: See the Notes section of
:func:`.convert_data_to_3d_array_layout` for extra information on RGB(A)
array layouts.

**Outputting RGB(A)**: See the Notes section of
:meth:`.OffAxisHologram.get_data_with_input_layout` or
:meth:`.QLSInterferogram.get_data_with_input_layout` for information on
outputting of RGB(A) array layouts.


Converting to and from Array Layouts
------------------------------------

`qpretrieve` will automatically handle the above allowed array layouts.
In other words, if you provide any 2D, RGB, RGBA, or 3D data as input to
:class:`.OffAxisHologram` or :class:`.QLSInterferogram`
the class will handle everything for you.

However, if you want to have your processed data in the same array layout as when
you inputted it, then you can use the convenience method
:meth:`get_data_with_input_layout` to do just that. For example, if
your input data was a 2D array, you can get the processed field, phase,
amplitude etc like so:

.. code-block:: python

	# 2D data inputted
	oah = qpretrieve.OffAxisHologram(hologram_2d)
	# do some processing
	...
	# get your data as a 2D array layout
	field_2d = oah.get_data_with_input_layout("field")
	phase_2d = oah.get_data_with_input_layout("phase")
	amplitude_2d = oah.get_data_with_input_layout("amplitude")

	# you can also use the class attributes
	field_2d = oah.get_data_with_input_layout(oah.field)
	phase_2d = oah.get_data_with_input_layout(oah.phase)
	amplitude_2d = oah.get_data_with_input_layout(oah.amplitude)
