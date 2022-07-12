#!/usr/bin/env python

"""UTILS.PY - Utility functions

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20220711'  # yyyymmdd                                                                                                                           

import os
import numpy as np
import warnings
from astropy.io import fits
from astropy.table import Table
from dlnpyutils import utils as dln
from gaiaxpy import calibrate
import pandas as pd
try:
    import __builtin__ as builtins # Python 2
except ImportError:
    import builtins # Python 3
        
# Ignore these warnings, it's a bug
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")


def datadir():
    """ Return the doppler data/ directory."""
    fil = os.path.abspath(__file__)
    codedir = os.path.dirname(fil)
    datadir = codedir+'/data/'
    return datadir

# Split a filename into directory, base and fits extensions
def splitfilename(filename):
    """ Split filename into directory, base and extensions."""
    fdir = os.path.dirname(filename)
    base = os.path.basename(filename)
    exten = ['.fit','.fits','.fit.gz','.fits.gz','.fit.fz','.fits.fz']
    for e in exten:
        if base[-len(e):]==e:
            base = base[0:-len(e)]
            ext = e
            break
    return (fdir,base,ext)

def getprintfunc(inplogger=None):
    """ Allows you to modify print() locally with a logger."""
    
    # Input logger
    if inplogger is not None:
        return inplogger.info  
    # Check if a global logger is defined
    elif hasattr(builtins,"logger"):
        return builtins.logger.info
    # Return the buildin print function
    else:
        return builtins.print


def normalizeflux(flux,flux_error):
    """ Normalize a Bp/Rp flux spectrum."""

    # normalize coefficients
    if flux.ndim==2:
        medflux = np.nanmedian(flux,axis=1)
        newflux = flux / (medflux.reshape(-1,1) + np.zeros(343,float).reshape(1,-1))
        newflux_error = flux_error / (medflux.reshape(-1,1) + np.zeros(343,float).reshape(1,-1))
    else:
        medflux = np.nanmedian(flux)
        newflux = flux/medflux
        newflux_error = flux_error/medflux
    return newflux, newflux_error, medflux

    
def normalize(tab,fluxnorm=False):
    if 'flux' in tab.columns:
        newflux, newflux_error, medflux = normalizeflux(tab['flux'],tab['flux_error'])
        tab['flux'] = newflux
        tab['flux_error'] = newflux_error
        tab['medflux'] = medflux
        return tab
    if 'bp_coefficients' in tab.columns:
        mnfluxbases = np.mean(fluxbases,axis=2)
        # if fluxnorm is set then the mean of the flux will be 1.0
        if fluxnorm==False:
            mnfluxbases /= np.mean(mnfluxbases)
        bpmnfluxbases = mnfluxbases[0,:]
        rpmnfluxbases = mnfluxbases[1,:]
        # Calculating the normalization
        mn = np.sum(tab['bp_coefficients']*bpmnfluxbases + tab['rp_coefficients']*rpmnfluxbases,axis=1)
        tab['coeff_norm'] = mn
        tab['bp_coefficients'] /= mn.reshape(-1,1)
        tab['bp_coefficient_errors'] /= mn.reshape(-1,1)
        tab['rp_coefficients'] /= mn.reshape(-1,1)
        tab['rp_coefficient_errors'] /= mn.reshape(-1,1)
        return tab

def truncate(tab,nbp=None,nrp=None):
    """ Truncate coefficients."""
    if 'bp_n_relevant_bases' not in tab.columns or 'rp_n_relevant_bases' not in tab.columns:
        raise ValueError('table must have bp_n_relevant_bases and rp_n_relevant_bases')
    for i in range(len(tab)):
        if nbp is None:
            numbp = int(tab['bp_n_relevant_bases'][i])
        else:
            numbp = nbp
            tab['bp_n_relevant_bases'][i] = nbp
        if numbp < 55:
            tab['bp_coefficients'][i,numbp:] = 0.0
            tab['bp_coefficient_errors'][i,numbp:] = 1e30
        if nrp is None:
            numrp = int(tab['bp_n_relevant_bases'][i])
        else:
            numrp = nrp
            tab['rp_n_relevant_bases'][i] = nrp
        if numrp < 55:
            tab['rp_coefficients'][i,numrp:] = 0.0
            tab['rp_coefficient_errors'][i,numrp:] = 1e30
    return tab
        
def getflux(tab):
    """ Calculate flux from coefficients."""

    # columns it needs in dataframe: source_id, bp_n_parameters, bp_standard_deviation, bp_coefficients, bp_coefficient_errors,
    #                          bp_coefficient_correlations, same for rp
    
    # Convert to df if necessary
    if type(tab) != pd.DataFrame:
        dt = [('source_id',int),('bp_n_parameters',int),('bp_standard_deviation',float),
              ('bp_coefficients',object),('bp_coefficient_errors',object),('bp_coefficient_correlations',object),
              ('rp_n_parameters',int),('rp_standard_deviation',float),('rp_coefficients',object),
              ('rp_coefficient_errors',object),('rp_coefficient_correlations',object)]
        temp = np.zeros(len(tab),dtype=np.dtype(dt))
        if 'source_id' in tab.columns:
            temp['source_id'] = tab['source_id']
        else:
            temp['source_id'] = np.arange(len(tab))+1
        if 'bp_n_parameters' in tab.columns:
            temp['bp_n_parameters'] = tab['bp_n_parameters']
        else:
            temp['bp_n_parameters'] = 55
        if 'bp_standard_deviation' in tab.columns:
            temp['bp_standard_deviation'] = tab['bp_standard_deviation']
        else:
            temp['bp_standard_deviation'] = 1.0
        if 'rp_n_parameters' in tab.columns:
            temp['rp_n_parameters'] = tab['rp_n_parameters']
        else:
            temp['rp_n_parameters'] = 55
        if 'rp_standard_deviation' in tab.columns:
            temp['rp_standard_deviation'] = tab['rp_standard_deviation']
        else:
            temp['rp_standard_deviation'] = 1.0
        # xp_coefficients_correlation has 1485 elements            
        blank1 = '('+54*'0.0,'+'0.0)'
        blank2 = '('+1484*'0.0,'+'0.0)'        
        for i in range(len(tab)):
            temp['bp_coefficients'][i] = '('+','.join(np.char.array(tab['bp_coefficients'][i]).astype(str))+')'
            temp['rp_coefficients'][i] = '('+','.join(np.char.array(tab['rp_coefficients'][i]).astype(str))+')'
            if 'bp_coefficient_errors' in tab.columns:
                temp['bp_coefficient_errors'][i] = '('+','.join(np.char.array(tab['bp_coefficient_errors'][i]).astype(str))+')'
            else:
                temp['bp_coefficient_errors'][i] = blank1
            if 'rp_coefficient_errors' in tab.columns:
                temp['rp_coefficient_errors'][i] = '('+','.join(np.char.array(tab['rp_coefficient_errors'][i]).astype(str))+')'
            else:
                temp['rp_coefficient_errors'][i] = blank1
            if 'bp_coefficient_correlations' in tab.columns:
                temp['bp_coefficient_correlations'][i] = '('+','.join(np.char.array(tab['bp_coefficient_correlations'][i]).astype(str))+')'
            else:
                temp['bp_coefficient_correlations'][i] = blank2
            if 'rp_coefficient_correlations' in tab.columns:
                temp['rp_coefficient_correlations'][i] = '('+','.join(np.char.array(tab['rp_coefficient_correlations'][i]).astype(str))+')'
            else:
                temp['rp_coefficient_correlations'][i] = blank2               

        # Convert convert to pandas data frame  
        df = pd.DataFrame(temp)

    # Already pandas dataframe
    #   check that we have all of the needed columns
    else:
        df = tab
        # xp_coefficients_correlation has 1485 elements
        blank1 = np.array('('+54*'0.0,'+'0.0)',dtype=object)
        blank2 = np.array('('+1484*'0.0,'+'0.0)',dtype=object)        
        if 'source_id' not in df.columns:
            df['source_id'] = 0
            df['source_id'] = np.arange(len(tab))+1
        if 'bp_n_parameters' not in df.columns:
            df['bp_n_parameters'] = 55
        if 'bp_standard_deviation' not in df.columns:
            df['bp_standard_deviation'] = 1.0            
        if 'bp_coefficient_errors' not in df.columns:
            df['bp_coefficient_errors'] = blank1
        if 'bp_coefficient_correlations' not in df.columns:
            df['bp_coefficient_correlations'] = blank2      
        if 'rp_n_parameters' not in df.columns:
            df['rp_n_parameters'] = 55                        
        if 'rp_standard_deviation' not in df.columns:
            df['rp_standard_deviation'] = 1.0            
        if 'rp_coefficient_errors' not in df.columns:
            df['rp_coefficient_errors'] = blank1
        if 'rp_coefficient_correlations' not in df.columns:
            df['rp_coefficient_correlations'] = blank2            

    # Convert to flux
    calibrated_spectrum, sampling = calibrate(df)

    
    # If input was NOT pandas dataframe, then add to astropy table
    if type(tab) != pd.DataFrame:
        output = np.zeros(len(tab),dtype=np.dtype([('source_id',int),('flux',float,343),('flux_error',float,343)]))
        output = Table(output)
        output['source_id'] = calibrated_spectrum['source_id']
        for i in range(len(tab)):
            output['flux'][i,:] = np.array(calibrated_spectrum['flux'][i])
            output['flux_error'][i,:] = np.array(calibrated_spectrum['flux_error'][i])            
        return output

    return calibrated_spectrum
        

def load_library(libraryfile):
    """ Load the APOGEE library data and normalize"""
    print('Loading APOGEE Library file ',libraryfile)
    apogee = Table.read(libraryfile)
    # normalize the spectra
    newflux, newflux_error, medflux = normalize(apogee['flux'],apogee['flux_error'])
    apogee['medflux'] = medflux
    apogee['flux'] = newflux
    apogee['flux_error'] = newflux_error
    return apogee

fluxbases = fits.getdata(datadir()+'gaiaxp_bases.fits.gz',1)
