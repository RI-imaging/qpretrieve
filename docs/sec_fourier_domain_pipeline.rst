.. _sec_fourier_domain_pipeline:

=======================
Fourier Domain Pipeline
=======================

Since version 0.7.0, :meth:`.OffAxisHologram.run_pipeline` accepts an
``output_domain`` keyword argument. By default (``output_domain="spatial"``) the
pipeline returns the reconstructed complex field as usual. When
``output_domain="fourier"`` is set, the inverse FFT is skipped and a
:class:`~qpretrieve.fourier.fourier_field_data.FourierFieldData` is returned instead.

The :class:`~qpretrieve.fourier.fourier_field_data.FourierFieldData` holds the
filtered Fourier data and all the metadata needed to reconstruct the spatial field.
Calling its :meth:`~qpretrieve.fourier.fourier_field_data.FourierFieldData.finalize`
method performs the inverse FFT and returns the spatial field identically to the
default path.

.. admonition:: Combining `qpretrieve` and `nrefocus` pipelines

    The Fourier output is most useful when the result is passed directly to a
    wave propagation library such as `nrefocus
    <https://nrefocus.readthedocs.io>`_, which can consume the
    :class:`~qpretrieve.fourier.fourier_field_data.FourierFieldData` object and
    skip its own forward FFT. This avoids a redundant iFFT + FFT pair at the
    qpretrieve/nrefocus boundary.
    For an nrefocus-integrated working example see the :ref:`sec_examples`.

    *Spatial vs. Fourier inconsistency*

    For unpadded, square spatial input data, the default spatial domain
    pipeline and the fourier domain pipeline are identical. There is only
    floating point imprecision.
    For padded pipelines, the pipelines are not identical due to padding and
    unpadding causing inconsistencies at the boundary of the images.


Default: spatial output
-----------------------

.. code-block:: python

    import numpy as np
    import qpretrieve

    edata = np.load("examples/data/hologram_cell.npz")
    oah = qpretrieve.OffAxisHologram(edata["data"])
    field = oah.run_pipeline()      # returns complex spatial field
    print(type(field))              # numpy.ndarray
    print(field.shape)              # (1, H, W)

Fourier output: skip the inverse FFT
--------------------------------------

This is useful when combined with field propagation, see the Note above.

.. code-block:: python

    import numpy as np
    import qpretrieve

    edata = np.load("examples/data/hologram_cell.npz")
    oah = qpretrieve.OffAxisHologram(edata["data"])
    fourier_data = oah.run_pipeline(output_domain="fourier")

    # The fourier_data carries the filtered Fourier data.
    # Call finalize() to recover the spatial field when needed.
    field = fourier_data.finalize()
    print(field.shape)              # (1, H, W) — identical to the default path

