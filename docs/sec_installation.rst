Installing qpretrieve
=====================

qpretrieve is written in pure Python and supports Python version 3.6
and higher. qpretrieve depends on several other scientific Python packages,
including:

 - `numpy <https://docs.scipy.org/doc/numpy/>`_,
 - `scipy <https://docs.scipy.org/doc/scipy/reference/>`_, and
 - `scikit-image <http://scikit-image.org/>`_ (phase unwrapping using :py:func:`skimage.restoration.unwrap_phase`).
    

To install qpretrieve, use one of the following methods
(package dependencies will be installed automatically):
    
* from `PyPI <https://pypi.python.org/pypi/qpretrieve>`_:
    ``pip install qpretrieve``
* from `sources <https://github.com/RI-imaging/qpretrieve>`_:
    ``pip install -e .`` or
