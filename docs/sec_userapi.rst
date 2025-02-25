User API
========

The qpretrieve library offers user-convenient classes for
:ref:`QLSI <sec_code_qlsi>` and :ref:`DHM <sec_code_dhm>` data.

Please also check out the :ref:`examples section <sec_examples>`.

Let's say you have your DHM data loaded into a numpy array called ``hologram``.
Then your analysis could be as simple as

.. code-block:: python

   import qpretrieve
   import matplotlib.pylab as plt

   dhm = qpretrieve.OffAxisHologram(hologram)
   plt.imshow(dhm.phase)
   plt.show()

With ``dhm``, an instance of :class:`.OffAxisHologram`, you now have full access
to all intermediate computation results. You can pass additional keyword
arguments during instantiation or pass them to
:meth:`.OffAxisHologram.run_pipeline`.
