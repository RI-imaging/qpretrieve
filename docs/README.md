qpretrieve documentation
=====================
To install the requirements for building the documentation, run

    pip install -r requirements.txt

To compile the documentation, run

    sphinx-build . _build


Notes
=====
To view the sphinx inventory of qpretrieve, run

   python -m sphinx.ext.intersphinx 'http://qpretrieve.readthedocs.io/en/latest/objects.inv'
