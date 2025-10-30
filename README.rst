qpretrieve
==========

|PyPI Version| |Tests Status| |Coverage Status| |Docs Status|


**qpretrieve** is a Python3 library for Fourier-based retrieval of quantitative
phase information from experimental data. If you are looking for a library to
load quantitative phase data from various file formats, please take a glance at
`qpformat <https://github.com/RI-imaging/qpformat>`__.


Documentation
-------------

The documentation, including the code reference and examples, is available at
`qpretrieve.readthedocs.io <https://qpretrieve.readthedocs.io/en/stable/>`__.


Installation
------------

::

    pip install qpretrieve

For information on optional dependencies (FFTW, CuPy) see the
`installation guide <https://qpretrieve.readthedocs.io/en/stable/sec_installation.html>`__

Testing
-------

::

    pip install pytest
    pip install -e .
    pytest tests


.. |PyPI Version| image:: https://img.shields.io/pypi/v/qpretrieve.svg
   :target: https://pypi.python.org/pypi/qpretrieve
.. |Tests Status| image:: https://img.shields.io/github/actions/workflow/status/RI-Imaging/qpretrieve/check.yml
   :target: https://github.com/RI-Imaging/qpretrieve/actions?query=workflow%3AChecks
.. |Coverage Status| image:: https://img.shields.io/codecov/c/github/RI-imaging/qpretrieve/master.svg
   :target: https://codecov.io/gh/RI-imaging/qpretrieve
.. |Docs Status| image:: https://readthedocs.org/projects/qpretrieve/badge/?version=latest
   :target: https://readthedocs.org/projects/qpretrieve/builds/

