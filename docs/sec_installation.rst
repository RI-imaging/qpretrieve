Installing qpretrieve
=====================

qpretrieve is written in pure Python and supports Python version 3.10
and higher. qpretrieve depends on several other scientific Python packages,
including:

 - `numpy <https://docs.scipy.org/doc/numpy/>`_,
 - `scipy <https://docs.scipy.org/doc/scipy/reference/>`_, and
 - `scikit-image <http://scikit-image.org/>`_ (phase unwrapping using :py:func:`skimage.restoration.unwrap_phase`).
    
Base Package
------------

To install qpretrieve, use one of the following methods
(mandatory package dependencies will be installed automatically):
    
* from `PyPI <https://pypi.python.org/pypi/qpretrieve>`_:
    ``pip install qpretrieve``
* from `sources <https://github.com/RI-imaging/qpretrieve>`_:
    ``pip install -e .``

Optional Dependencies
---------------------

``qpretrieve`` has two optional dependencies:
`PyFFTW <https://pyfftw.readthedocs.io/en/latest/>`_ and
`CuPy <https://cupy.dev/>`_.

They can be installed individually or all at once:

   ``pip install qpretrieve[FFTW]``

   ``pip install qpretrieve[FFTW,CUPY]``


.. admonition:: CuPy & CUDA Versions

	CuPy works with CUDA devices. Running ``pip install qpretrieve[CUPY]`` will install
	``cupy-cuda12x``, which is compatible with CUDA version 12.
	If you have an older CUDA version, you will need to install
	``cupy-cuda11x`` and run ``pip install cupy-cuda11x`` in your environment.
	See the `CuPy website <https://cupy.dev/>`_ for details.
