import atexit
import shutil
import tempfile
import time

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
