from os.path import dirname, realpath, exists
from setuptools import setup, find_packages
import sys


author = u"Paul Müller"
authors = [author]
description = 'library for phase retrieval from holograms'
name = 'qpretrieve'
year = "2022"

sys.path.insert(0, realpath(dirname(__file__)) + "/" + name)
from _version import version  # noqa: E402

setup(
    name=name,
    author=author,
    author_email='dev@craban.de',
    url='https://github.com/RI-imaging/qpretrieve',
    version=version,
    packages=find_packages(),
    package_dir={name: name},
    license="MIT",
    description=description,
    long_description=open('README.rst').read() if exists('README.rst') else '',
    install_requires=[
        "numpy>=1.9.0",
        "scikit-image>=0.11.0",
        "scipy>=0.18.0",
        ],
    extras_require={"FFTW": "pyfftw>=0.12.0"},
    python_requires='>=3.6, <4',
    keywords=["digital holographic microscopy",
              "optics",
              "quantitative phase imaging",
              "refractive index",
              "numerical focusing",
              "scattering",
              ],
    classifiers=[
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Intended Audience :: Science/Research'
                 ],
    platforms=['ALL'],
    )
