#!/usr/bin/env python

"""SYNTH.PY - Create synthetic Bp/Rp spectra

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20220624'  # yyyymmdd                                                                                                                           

import os
import numpy as np
import warnings
from scipy import sparse
from scipy.interpolate import interp1d
from dlnpyutils import utils as dln
from gaiaxpy import calibrate, convert
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
        
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


# 1) Convolve with separate Bp and Rp resolution vs. wavelength
# 2) Decompose into Hermite polynomials
# 3) Use gaiaxpy calibrate() to create fluxed, combined spectrum

def synth(spec):
    """
    Create synthetic Bp/Rp spectra.
    """

    # Convolve
    cspec = convolve_spec(spec)

    # Decompose into Hermite polynomial coefficients
    hspec = decompose_spec(cspec)

    # Combine
    fspec = calibrate(hspec)

    # Do we need a throughput correction!!??
    
    return fpsec


def convolve_spec(spec,gcoef):
    """
    Convolve spectrum with a wavelength dependence smoothing function.
    """
    
    pass

def decompose_spec(spec):
    """
    Decompose a spectrum into Hermite polynomical coefficients.
    """

    # Use the calibrate() function to get the hermite functions
    
    pass

