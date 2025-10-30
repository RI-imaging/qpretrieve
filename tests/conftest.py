import atexit
import shutil
import tempfile
import time
import numpy as np

import pytest

import qpretrieve

TMPDIR = tempfile.mkdtemp(prefix=time.strftime(
    "qpretrieve_test_%H.%M_"))


def pytest_configure(config):
    """
    Allows plugins and conftest files to perform initial configuration.
    This hook is called for every plugin and initial conftest
    file after command line options have been parsed.
    """
    tempfile.tempdir = TMPDIR
    atexit.register(shutil.rmtree, TMPDIR, ignore_errors=True)
    # This will make the tests pass faster, because we are not
    # creating FFTW wisdom. Also, it makes the tests more reproducible
    # by sticking to simple numpy FFTs.
    qpretrieve.fourier.PREFERRED_INTERFACE = "FFTFilterNumpy"


@pytest.fixture(params=[64])  # default param for size
def hologram(request):
    size = request.param
    x = np.arange(size).reshape(-1, 1) - size / 2
    y = np.arange(size).reshape(1, -1) - size / 2

    amp = np.linspace(.9, 1.1, size * size).reshape(size, size)
    pha = np.linspace(0, 2, size * size).reshape(size, size)

    rad = x ** 2 + y ** 2 > (size / 3) ** 2
    pha[rad] = 0
    amp[rad] = 1

    # frequencies must match pixel in Fourier space
    kx = 2 * np.pi * -.3
    ky = 2 * np.pi * -.3
    image = (amp ** 2 + np.sin(kx * x + ky * y + pha) + 1) * 255
    return image
