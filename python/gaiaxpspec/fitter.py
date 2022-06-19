#!/usr/bin/env python

"""FITTING.PY - Routines to fit models to the Gaia Bp/Rp spectra.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20220618'  # yyyymmdd

import os
import numpy as np
import warnings
from glob import glob
from astropy.table import Table
from scipy.optimize import curve_fit
from dlnpyutils import (utils as dln, bindata, astro)
from .xpspec import XPSpec
from . import utils, model
import copy
import logging
import contextlib, io, sys
import time
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
    
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Get print function to be used locally, allows for easy logging
print = utils.getprintfunc() 

apogee = Table.read(utils.datadir()+'apogee_library_spectra.fits.gz')

# Load the default Payne model
def apogee_estimate(spec):
    """
    Figure out the best parameters of a Bp/RP spectrum using the APOGEE library.
    """

    # Initialize the output table
    dt = [('best_source_id',int),('best_apogee_id',(str,30)),('chisq',np.float32),
          ('teff',np.float32),('logg',np.float32),('feh',np.float32),('ak',np.float32)]
    out = np.zeros(1,dtype=np.dtype(dt))
    out = Table(out)

    # Calculate chisq
    wtresid = ( (apogee['flux']-spec.flux.reshape(1,-1))/spec.err.reshape(1,-1) )**2
    chisq = np.sum(wtresid,axis=1)
    bestind = np.argmin(chisq)
    bestflux = apogee['flux'][bestind]
    # Fit the table
    out['best_source_id'] = apogee['gaiaedr3_source_id'][bestind]
    out['best_apogee_id'] = apogee['apogee_id'][bestind]
    out['teff'] = apogee['teff'][bestind]
    out['logg'] = apogee['logg'][bestind]
    out['feh'] = apogee['fe_h'][bestind]
    out['ak'] = apogee['ak_targ'][bestind]
    out['chisq'] = chisq[bestind]

    return out

def solvestar(spec,initpar=None):
    """
    Solve the stellar parameters and extinction for a star.
    """

    # Get initial guess, if necessary
    if initpar is None:
        out = apogee_estimate(spec)
        initpar = [out['teff'][0],out['logg'][0],out['feh'][0],out['ak'][0]*9.25]
        if np.isfinite(out['feh'][0])==False:
            initpar[2] = 0.0

    # Load the model
    m = model.load_model()
    xpm = model.XPModel(m)
            
    # Use curve_fit to find the best values
    tol = 5e-4   # tolerance for when the optimizer should stop optimizing.

    # prohibit the minimimizer to go outside the range of training set
    bounds = np.zeros((2,4))
    bounds[0,0] = 3100   # teff
    bounds[1,0] = 6800
    bounds[0,1] = -0.3   # logg
    bounds[1,1] = 5.3
    bounds[0,2] = -2.4   # feh
    bounds[1,2] = 0.5
    bounds[0,3] = 0      # a55
    bounds[1,3] = 10

    def fit_func(x,*args):
        labels = args
        msp = xpm(labels)
        return msp.flux
    
    # run the optimizer
    pars, pcov = curve_fit(fit_func, xdata=[], ydata = spec.flux, sigma = spec.err, p0 = initpar,
                           bounds = bounds, ftol = tol, xtol = tol, absolute_sigma = True, method = 'trf')
    perror = np.sqrt(np.diag(pcov))
    mspec = xpm(pars)
    chisq = np.sum( (spec.flux-mspec.flux)**2 / spec.err**2 )

    # MCMC
    
    
    # Initialize the output table
    dt = [('teff',np.float32),('teff_err',np.float32),('logg',np.float32),('logg_err',np.float32),
          ('feh',np.float32),('feh_err',np.float32),('a55',np.float32),('a55_err',np.float32),
          ('chisq',np.float32),]
    out = np.zeros(1,dtype=np.dtype(dt))
    out = Table(out)
    out['teff'] = pars[0]
    out['teff_err'] = perror[0]    
    out['logg'] = pars[1]
    out['logg_err'] = perror[1]        
    out['feh'] = pars[2]
    out['feh_err'] = perror[2]        
    out['a55'] = pars[3]
    out['a55_err'] = perror[3]    
    out['chisq'] = chisq

    return out, mspec
    
