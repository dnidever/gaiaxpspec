#!/usr/bin/env python

"""MODEL.PY - Routines to work with Payne models.

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20220618'  # yyyymmdd

# Must of this is copied from doppler/payne.py

# Some of the software is from Yuan-Sen Ting's The_Payne repository
# https://github.com/tingyuansen/The_Payne

import os
import numpy as np
import warnings
from glob import glob
from astropy.table import Table
from scipy.interpolate import interp1d
from dlnpyutils import (utils as dln, bindata, astro)
from .xpspec import XPSpec
from . import utils
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

cspeed = 2.99792458e5  # speed of light in km/s

# Get print function to be used locally, allows for easy logging
print = utils.getprintfunc() 

def leaky_relu(z):
    """ This is the activation function used by default in all our neural networks. """
    return z*(z > 0) + 0.01*z*(z < 0)

# Load the default Payne model
def load_model():
    """
    Load the default Payne model.
    """

    datadir = utils.datadir()
    files = glob(datadir+'payne_cool_gaiaxp.npz')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No model files in "+datadir)
    if nfiles>1:
        return ModelSet.read(files)
    else:
        return Model.read(files[0])


# Load a single or list of Payne models
def load_payne_model(mfile):
    """
    Load a  Payne model from file.

    Returns
    -------
    mfiles : string
       File name (or list of filenames) of Payne models to load.

    Examples
    --------
    model = load_payne_model()

    """
    
    if os.path.exists(mfile) == False:
        raise ValueError(mfile+' not found')
    
    # read in the weights and biases parameterizing a particular neural network. 
    tmp = np.load(mfile)
    w_array_0 = tmp["w_array_0"]
    w_array_1 = tmp["w_array_1"]
    w_array_2 = tmp["w_array_2"]
    b_array_0 = tmp["b_array_0"]
    b_array_1 = tmp["b_array_1"]
    b_array_2 = tmp["b_array_2"]
    x_min = tmp["x_min"]
    x_max = tmp["x_max"]
    if 'wavelength' in tmp.files:
        wavelength = tmp["wavelength"]
    else:
        # 336 - 1020 in steps of 2 nm
        wavelength = np.arange(336,1022,2)*10.0
    #if 'labels' in tmp.files:
    #    labels = list(tmp["labels"])
    #else:
    #    print('WARNING: No label array')
    #    labels = [None] * w_array_0.shape[1]
    labels = ['TEFF','LOGG','FEH']
    coeffs = (w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max)
    tmp.close()
    
    return coeffs, wavelength, labels

def load_models():
    """
    Load all Payne models from the Doppler data/ directory
    and return as a DopplerPayneModel.

    Returns
    -------
    models : DopplerPayneModel
        DopplerPayneModel for all Payne models in the
        Doppler /data directory.

    Examples
    --------
    models = load_models()

    """    
    datadir = utils.datadir()
    files = glob(datadir+'payne_coolhot_*.npz')
    nfiles = len(files)
    if nfiles==0:
        raise Exception("No Payne model files in "+datadir)
    return DopplerPayneModel.read(files)

def check_params(model,params):
    """ Check input fit or fixed parameters against Payne model labels."""
    # Check the input labels against the Paybe model labels

    if isinstance(params,dict):
        paramdict = params.copy()
        params = list(paramdict.keys())
        isdict = True
    else:
        isdict = False

    # Check for duplicates
    uparams = np.unique(np.array(params))
    if len(uparams)!=len(params):
        raise ValueError('There are duplicates in '+','.join(params))
        
    # Loop over parameters
    for i,par in enumerate(params):
        # replace VROT with VSINI        
        if par=='VROT' and 'VSINI' in model.labels:
            print('Replacing VROT -> VSINI')
            params[i] = 'VSINI'
            par = 'VSINI'
        # replace VMICRO with VTURB            
        elif par=='VMICRO' and 'VTURB' in model.labels:
            print('Replacing VMICRO -> VTURB')
            params[i] = 'VTURB'
            par = 'VTURB'
        # check against model labels
        if (par != 'ALPHA_H') and (not par in model.labels):
            raise ValueError(par+' NOT a Payne label. Available labels are '+','.join(model.labels)+' and ALPHA_H')

    # Return "adjusted" params
    if isdict==True:
        paramdict = dict(zip(params,paramdict.values()))
        return paramdict
    else:    
        return params
    
def mkdxlim(fitparams):
    """
    Make array of parameter changes at which curve_fit should finish.

    Parameters
    ----------
    fitparams : list
        List of parameter names.

    Returns
    -------
    dx_lim : numpy array
       Array of parameter changes at which curve_fit should finish.

    Example
    -------
    .. code-block:: python

         dx_lim = mkdxlim(fitparams)

    """
    npar = len(fitparams)
    dx_lim = np.zeros(npar,float)
    for k in range(npar):
        if fitparams[k]=='TEFF':
            dx_lim[k] = 1.0
        elif fitparams[k]=='LOGG':
            dx_lim[k] = 0.005
        elif (fitparams[k]=='VMICRO' or fitparams[k]=='VTURB'):
            dx_lim[k] = 0.1
        elif (fitparams[k]=='VSINI' or fitparams[k]=='VROT'):
            dx_lim[k] = 0.1
        elif fitparams[k]=='VMACRO':
            dx_lim[k] = 0.1
        elif fitparams[k]=='RV':
            dx_lim[k] = 0.01
        elif fitparams[k].endswith('_H'):
            dx_lim[k] = 0.005
        else:
            dx_lim[k] = 0.01
    return dx_lim

def mkinitlabels(labels):
    """
    Make initial guesses for Payne labels.

    Parameters
    ----------
    labels : list
        List of parameter names.

    Returns
    -------
    initpars : numpy array
       Array of initial label values.

    Example
    -------
    .. code-block:: python

         initpars = mkinitlabels(labels)

    """

    labels = np.char.array(labels).upper()
    
    # Initializing the labels array
    nlabels = len(labels)
    initpars = np.zeros(nlabels,float)
    initpars[labels=='TEFF'] = 5000.0
    initpars[labels=='LOGG'] = 3.5
    initpars[labels.endswith('_H')] = 0.0
    # Vmicro/Vturb=2.0 km/s by default
    initpars[(labels=='VTURB') | (labels=='VMICRO')] = 2.0
    # All abundances, VSINI, VMACRO, RV = 0.0
            
    return initpars


def mkbounds(labels,initpars=None):
    """
    Make upper and lower bounds for Payne labels.

    Parameters
    ----------
    labels : list
        List of parameter names.
    initpars : numpy array, optional
        Input list of initial label guesses.  Optional

    Returns
    -------
    bounds : tuple
       Two-element tuple of lower and upper boundaries for the input labels.

    Example
    -------
    .. code-block:: python

         bounds = mkbounds(labels,initpars)

    """

    if initpars is None:
        initpars = mkinitlabels(labels)
    nlabels = len(labels)
    lbounds = np.zeros(nlabels,np.float64)
    ubounds = np.zeros(nlabels,np.float64)

    # Initial guesses and bounds for the fitted parameters
    for i,par in enumerate(labels):
        if par.upper()=='TEFF':
            lbounds[i] = np.maximum(initpars[i]-2000,3000)
            ubounds[i] = initpars[i]+2000
        if par.upper()=='LOGG':
            lbounds[i] = np.maximum(initpars[i]-2,0)
            ubounds[i] = np.minimum(initpars[i]+2,5)
        if par.upper()=='VMICRO' or par.upper()=='VTURB':
            lbounds[i] = np.maximum(initpars[i]-2,0)
            ubounds[i] = initpars[i]+2
        if par.upper().endswith('_H'):
            lbounds[i] = np.maximum(initpars[i]-0.75,-2.5)
            ubounds[i] = np.minimum(initpars[i]+0.75,0.5)
        if par.upper()=='FE_H':
            lbounds[i] = -2.5
            ubounds[i] = 0.5
        if par.upper()=='VSINI' or par.upper()=='VROT':
            lbounds[i] = np.maximum(initpars[i]-20,0)
            ubounds[i] = initpars[i]+50
        if par.upper()=='VMACRO':
            lbounds[i] = np.maximum(initpars[i]-2,0)
            ubounds[i] = initpars[i]+2
        if par.upper()=='RV':
            lbounds[i] = -1000.0
            ubounds[i] = 1000.0
            
    bounds = (lbounds,ubounds)
    
    return bounds


class Model(object):
    """
    A class to represent a Payne Artificial Neural Network model.

    Parameters
    ----------
    coeffs : list
        List of Payne coefficient arrays.
    wavelength : numpy array
        Array of wavelength values.
    labels : list
        List of Payne labels.

    """
    
    def __init__(self,coeffs,wavelength,labels):
        """ Initialize PayneModel object. """
        self._coeffs = coeffs
        self._dispersion = wavelength
        self.labels = list(labels)
        wr = np.zeros(2,np.float64)
        wr[0] = np.min(wavelength)
        wr[1] = np.max(wavelength)
        self.wr = wr
        self.npix = len(self._dispersion)

    def __repr__(self):
        """ String representation."""
        out = self.__class__.__name__
        out += '('+','.join(self.labels)+')\n'        
        return out

    def __str__(self):
        """ String representation."""
        out = self.__class__.__name__
        out += '('+','.join(self.labels)+')\n'
        return out   
        
    @property
    def dispersion(self):
        """ Wavelength array."""
        return self._dispersion

    @dispersion.setter
    def dispersion(self,disp):
        if len(disp) != len(self._dispersion):
            raise ValueError('Input dispersion array not of the right length')
        self._dispersion = disp

    def __call__(self,labels,spec=None,fluxonly=False,wave=None):
        """
        Create the Payne model spectrum given the input label values.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        spec : Spec1D object, optional
            Observed spectrum to use for LSF convolution and wavelength array.  Default is to return
            the full model spectrum with no convolution.

        Returns
        -------
        mspec : numpy array or Spec1D object
            The output model Payne spectrum.  If fluxonly=True then only the flux array is returned,
            otherwise a Spec1D object is returned.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """
        
        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')        

        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''

        # assuming your NN has two hidden layers.
        w_array_0, w_array_1, w_array_2, b_array_0, b_array_1, b_array_2, x_min, x_max = self._coeffs
        scaled_labels = (labels-x_min)/(x_max-x_min) - 0.5
        inside = np.einsum('ij,j->i', w_array_0, scaled_labels) + b_array_0
        outside = np.einsum('ij,j->i', w_array_1, leaky_relu(inside)) + b_array_1
        spectrum = np.einsum('ij,j->i', w_array_2, leaky_relu(outside)) + b_array_2

        # Return as spectrum object with wavelengths
        if fluxonly is False:
            mspec = XPSpec(spectrum,wave=self.dispersion,instrument='Model')
            mspec.labels = labels
            for i,l in enumerate(self.labels): setattr(mspec,l.lower(),labels[i])
        else:
            mspec = spectrum
                
        return mspec

    def copy(self):
        """ Make a full copy of the Model object. """
        new_coeffs = []
        for c in self._coeffs:
            new_coeffs.append(c.copy())
        new = Model(new_coeffs,self._dispersion.copy(),self.labels.copy())
        return new

    
    @classmethod
    def read(cls,mfile):
        """ Read in a single Payne Model."""
        coeffs, wavelength, labels = load_payne_model(mfile)
        return Model(coeffs, wavelength, labels)

   
class XPModel(object):
    """
    A class to thinly wrap the Model so that we can add extinction to it.
    This is the primary model class to use with gaiaxpspec.

    Parameters
    ----------
    models : Model
        Model object.
    """

    def __init__(self,model):
        """ Initialize Model object. """
        # Make sure it's a Model 
        if not isinstance(model,Model):
            raise ValueError("Model must be Model")
        self._data = model
        labels = list(model.labels.copy())
        # Add vsini, vmacro, rv
        labels += ['A55','R55']
        self.labels = labels
        self._spec = None
        self._dispersion = self._data.dispersion
        self.wr = self._data.wr
        
    def __call__(self,labels):
        """
        Create the model spectrum given the input label values.

        Parameters
        ----------
        labels : list or array
            List/array or dictionary of input labels values to use.

        Returns
        -------
        mspec : numpy array or XPSpec object
            The output model spectrum.  If fluxonly=True then only the flux array is returned,
            otherwise a XPSpec object is returned.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """
        # Dictionary input
        if isinstance(labels,dict):
            labels = self.mklabels(labels)  # convert dictionary to array of labels
            
        if len(labels) < 3:
            raise ValueError('labels must have 3 elements')
        if len(labels)==3:
            a55,r55 = 0.0,3.02
        elif len(labels)==4:
            a55 = labels[3]
            r55 = 3.02
        elif len(labels)==5:
            a55 = labels[3]
            r55 = labels[4]
        plabels = labels[0:3]  # just the payne labels
        # Get the model spectrum
        spec = self._data(plabels)

        # Save the original unextincted flux
        spec._flux = spec.flux.copy()
        # Normalize
        spec.normalize()
        
        # Apply the extinction
        spec.extinct(a55,r55)

        # Set the labels
        spec.labels = list(plabels)+[a55,r55]
        for i,l in enumerate(self.labels):
            setattr(spec,l.lower(),spec.labels[i])
        
        return spec
        
    def __repr__(self):
        """ String representation."""
        out = self.__class__.__name__
        out += '('+','.join(self.labels)+')\n'        
        return out

    def __str__(self):
        """ String representation."""
        out = self.__class__.__name__
        out += '('+','.join(self.labels)+')\n'
        return out   

    def extinct(self,A55,R55=3.02):
        """ Apply extinction to the spectrum."""
        flux = self._flux.copy()
        self.a55 = A55
        self.r55 = R55
        
        # Extinct it
        if a55 > 0:
            # Get the extinction            
            alambda,attenuation = fitz_extinct(a55,r55)
            # Now apply the extinction to the model
            self.flux = flux*attenuation
            # Normalize again
            self.normalize()
            
    @property
    def dispersion(self):
        """ Wavelength array."""
        return self._dispersion
    
    def copy(self):
        """ Make a copy of the DopplerPayneModel but point to the original data."""
        new_model = self._data.copy()
        new = XPModel(new_model)
        # points to the ORIGINAL Payne data to save space        
        new._data = self._data
        return new

    def hardcopy(self):
        """ Make a complete copy of the XPModel including the original data."""
        new_model = self._data.copy()
        new = XPModel(new_model)
        return new    

    @classmethod
    def read(cls,mfiles):
        """ Read a set of model files."""
        model = Model.read(mfiles)
        return XPModel(model)

    
class XPModelSet(object):
    """
    A class to represent a set of Payne Artificial Neural Network models.  This is used
    when separate Payne models are used to cover a different regions of parameter space.

    Parameters
    ----------
    models : list of Model objects
        List of Model objects.

    """
    
    def __init__(self,models):
        """ Initialize Model object. """
        # Make sure it's a list
        if type(models) is not list:
            models = [models]
        # Check that the input is Payne models
        if not isinstance(models[0],Model):
            raise ValueError('Input must be list of Model models')
            
        self.nmodel = len(models)
        self._data = models
        wrarray = np.zeros((2,len(models)),np.float64)
        disp = []
        for i in range(len(models)):
            wrarray[0,i] = np.min(models[i].dispersion)
            wrarray[1,i] = np.max(models[i].dispersion)
            disp += list(models[i].dispersion)
        self._wrarray = wrarray
        self._dispersion = np.array(disp)

        self.npix = len(self._dispersion)
        wr = np.zeros(2,np.float64)
        wr[0] = np.min(self._dispersion)
        wr[1] = np.max(self._dispersion)
        self.wr = wr   # global wavelength range
        self.labels = self._data[0].labels
        
    @property
    def dispersion(self):
        """ Wavelength array."""
        return self._dispersion

    def __call__(self,labels,fluxonly=False):
        """
        Create the Payne model spectrum given the input label values.

        Parameters
        ----------
        labels : list or array
            List or Array of input labels values to use.
        fluxonly : boolean, optional
            Only return the flux array.  Default is to return a XPSpec object.

        Returns
        -------
        mspec : numpy array or Spec1D object
            The output model Payne spectrum.  If fluxonly=True then only the flux array is returned,
            otherwise a Spec1D object is returned.

        Example
        -------
        .. code-block:: python

             mspec = model(labels)

        """

        
        '''
        Predict the rest-frame spectrum (normalized) of a single star.
        We input the scaled stellar labels (not in the original unit).
        Each label ranges from -0.5 to 0.5
        '''

        if len(labels) != len(self.labels):
            raise ValueError('labels must have '+str(len(self.labels))+' elements')

        
        # This will return a model given a set of parameters
        if pars is None:
            pars = np.array([teff,logg,feh])
            pars = pars.flatten()
        # Get best model
        model = self.get_best_model(pars)
        if model is None: return None

        return model(pars,order=order,norm=norm,fluxonly=fluxonly,wave=wave,rv=rv)

    def get_best_model(self,pars):
        """ This returns the first XPModel instance that has the right range."""
        for m in self._data:
            ranges = m.ranges
            inside = True
            # Only search in Teff and logg
            for i in range(2):
                inside &= (pars[i]>=ranges[i,0]) & (pars[i]<=ranges[i,1])
            if inside:
                return m
        return None
    
    def __repr__(self):
        """ String representation."""
        out = self.__class__.__name__+'\n'
        out += ','.join(str.labels)
        return out

    def __str__(self):
        """ String representation."""
        out = self.__class__.__name__+'\n'
        out += ','.join(str.labels)
        return out     
    
    def __setitem__(self,index,data):
        self._data[index] = data
    
    def __getitem__(self,index):
        # Return one of the Payne models in the set
        return self._data[index]

    def __len__(self):
        return self.nmodel
    
    def __iter__(self):
        self._count = 0
        return self
        
    def __next__(self):
        if self._count < self.nmodel:
            self._count += 1            
            return self._data[self._count-1]
        else:
            raise StopIteration

    def copy(self):
        """ Make a copy of the PayneModelSet."""
        new_models = []
        for d in self._data:
            new_models.append(d.copy())
        new = ModelSet(new_models)
        return new

    @classmethod
    def read(cls,mfiles):
        """ Read a set of Payne model files."""
        n = len(mfiles)
        models = []
        for i in range(n):
            models.append(PayneModel.read(mfiles[i]))
        # Sort by wavelength
        def minwave(m):
            return m.dispersion[0]
        models.sort(key=minwave)
        return PayneModelSet(models)



class SpecFitter:
    """
    This is a special class that helps with least-squares fitting of a Payne model
    to an observed spectrum using functions like curve_fit().

    Parameters
    ----------
    spec : Spec1D object
       Observed spectrum to fit.
    pmodel : Model object
       The Model object to use for the model.
    fitparams : list, optional
       List of label names to fit.  Default is to fit all labels.
    fixparams : dictionary, optional
       Dictionary of parameter values to hold fixed.  Default is to not hold
       any values fixed.
    verbose : boolean, optional
       Verbose output to the screen.  Default is False.
    """
    
    def __init__(self,spec,pmodel,fitparams=None,fixparams={},verbose=False):
        """ Initialize SpecFitter object."""
        # spec - observed spectrum object
        # pmodel - Payne model object
        # params - initial/fixed parameters dictionary
        # fitparams - parameter/label names to fit (default is all)
        # "Prepare" the Payne model with the observed spectrum
        if pmodel.prepared is False: pmodel.prepare(spec)
        self._paynemodel = pmodel
        self.labels = pmodel.labels
        labelnames = np.char.array(self._paynemodel.labels)
        nlabels = len(self._paynemodel.labels)
        self.fixparams = dict((key.upper(), value) for (key, value) in fixparams.items()) # all CAPS
        self._initlabels = self.mkinitlabels(fixparams)
        if fitparams is not None:
            self.fitparams = fitparams
        else:
            self.fitparams = paynemodel.labels # by default fit all Payne parameters
        self._nfit = len(self.fitparams)

        # Labels FIXED, ALPHAELEM, ELEM arrays
        fixed = np.ones(nlabels,bool)  # all fixed by default
        alphaelem = np.zeros(nlabels,bool)  # all False to start
        elem = np.zeros(nlabels,bool)  # all False to start        
        for k,name in enumerate(labelnames):
            # Alpha element
            if name in ['O_H','MG_H','SI_H','S_H','CA_H','TI_H']:
                alphaelem[k] = True
                elem[k] = True
                # In FITPARAMS, NOT FIXED
                if name in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in FIXPARAMS, FIXED                    
                elif name in self.fixparams.keys():
                    fixed[k] = True
                # Not in FITPARAMS or FIXPARAMS, but FE_H or ALPHA_H in FITPARAMS, NOT FIXED
                elif 'FE_H' in self.fitparams or 'ALPHA_H' in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS/PARAMS and FE_H/ALPHA_H not being fit, FIXED
                else:
                    fixed[k] = True
            # Non-alpha element
            elif name.endswith('_H'):
                elem[k] = True
                # In FITPARAMS, NOT FIXED
                if name in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in FIXPARAMS, FIXED
                elif name in self.fixparams.keys():
                    fixed[k] = True
                # Not in FITPARAMS or FIXPARAMS, but FE_H in FITPARAMS, NOT FIXED
                elif 'FE_H' in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS/FIXPARAMS and FE_H not being fit, FIXED
                else:
                    fixed[k] = True
            # Other parameters (Teff, logg, RV, Vturb, Vsini, etc.)
            else:
                # In FITPARAMS, NOT FIXED
                if name in self.fitparams:
                    fixed[k] = False
                # Not in FITPARAMS but in FIXPARAMS, FIXED
                elif name in self.fixparams.keys():
                    fixed[k] = True
                # Not in FITPARAMS/PARAMS, FIXED
                else:
                    fixed[k] = True
        self._label_fixed = fixed
        self._label_alphaelem = alphaelem
        self._label_elem = elem        

        self._spec = spec.copy()
        self._flux = spec.flux.flatten()
        self._err = spec.err.flatten()
        self._wave = spec.wave.flatten()
        self._lsf = spec.lsf.copy()
        self._lsf.wavevac = spec.wavevac
        self._wavevac = spec.wavevac
        self.verbose = verbose
        #self._norm = norm   # normalize
        self._continuum_func = spec.continuum_func
        # Figure out the wavelength parameters
        npix = spec.npix
        norder = spec.norder
        # parameters to save
        self.nfev = 0
        self.njac = 0
        self._all_pars = []
        self._all_model = []
        self._all_chisq = []
        self._jac_array = None

    @property
    def fixparams(self):
        """ Dictionary of fixed parameters."""
        return self._fixparams

    @fixparams.setter
    def fixparams(self,fixparams):
        """ Dictionary, keys must be all CAPS."""
        self._fixparams = dict((key.upper(), value) for (key, value) in fixparams.items())  # all CAPS
            
    @property
    def fitparams(self):
        """ List of labels to fit."""
        return self._fitparams

    @fitparams.setter
    def fitparams(self,fitparams):
        """ List, all CAPS."""
        self._fitparams = [f.upper() for f in fitparams]

    def mkinitlabels(self,inputs):
        """
        Make initial guesses for Payne labels.

        Parameters
        ----------
        inputs : dict
           Dictionary of parameter values to use in the array.

        Returns
        -------
        labels : numpy array
           Array of initial label values.

        Example
        -------
        .. code-block:: python

             labels = spfitter.mkinitlabels(labeldict)
        
        """
        
        # This assumes ALL abundances are relative to H *not* FE!!!
        
        params = dict((key.upper(), value) for (key, value) in inputs.items()) # all CAPS
        nparams = len(params)

        labelnames = np.char.array(self._paynemodel.labels)

        # Defaults for main parameters
        if 'TEFF' not in list(params.keys()):
            params['TEFF'] = 4000.0
        if 'LOGG' not in list(params.keys()):
            params['LOGG'] = 3.0
        if 'FE_H' not in list(params.keys()):
            params['FE_H'] = 0.0            
            
            
        # Initializing the labels array
        nlabels = len(self._paynemodel.labels)
        labels = np.zeros(nlabels,float)
        # Set X_H = FE_H
        labels[labelnames.endswith('_H')] = params['FE_H']
        # Vmicro/Vturb=2.0 km/s by default
        labels[(labelnames=='VTURB') | (labelnames=='VMICRO')] = 2.0
        
        # Deal with alpha abundances
        # Individual alpha elements will overwrite the mean alpha below     
        # Make sure ALPHA_H is *not* one of the labels:
        if 'ALPHA_H' not in self._paynemodel.labels:
            if 'ALPHA_H' in params.keys():
                alpha = params['ALPHA_H']
                alphaelem = ['O','MG','SI','S','CA','TI']                
                for k in range(len(alphaelem)):
                    # Only set the value if it was found in self.labels
                    labels[labelnames==alphaelem[k]+'_H'] = alpha
                
        # Loop over input parameters
        for name in params.keys():
            # Only set the value if it was found in labelnames
            labels[labelnames==name] = params[name]
            
        return labels

        
    def mklabels(self,args):
        """
        Make labels for Payne model using values for only the fitted values.

        Parameters
        ----------
        args : list or tuple
           List or tuple of values for the fitted parameters (fitparams).

        Returns
        -------
        labels : numpy array
           Array of values for all the Payne labels.

        Example
        -------
        .. code-block:: python

             labels = spfitter.mklabels(args)
              
        """
        # Start with initial labels and only modify the fitparams."""

        # Initialize with init values
        labels = self._initlabels.copy()
        
        labelnames = np.char.array(self._paynemodel.labels)
        fitnames = np.char.array(self.fitparams)
        if 'FE_H' in self.fitparams:
            fitfeh = True
            fehind, = np.where(fitnames=='FE_H')
        else:
            fitfeh = False
        if 'ALPHA_H' in self.fitparams:
            fitalpha = True
            alphaind, = np.where(fitnames=='ALPHA_H')
        else:
            fitalpha = False

        # Loop over labels        
        for k,name in enumerate(labelnames):
            # Label is NOT fixed, change it
            if self._label_fixed[k] == False:
                # Alpha element
                if self._label_alphaelem[k] == True:
                    # ALPHA_H in FITPARAMS
                    if fitalpha is True:
                        labels[k] = args[alphaind[0]]
                    elif fitfeh is True:
                        labels[k] = args[fehind[0]]
                    else:
                        print('THIS SHOULD NOT HAPPEN!')
                        import pdb; pdb.set_trace()
                # Non-alpha element
                elif self._label_elem[k] == True:
                    if fitfeh is True:
                        labels[k] = args[fehind[0]]
                    else:
                        print('THIS SHOULD NOT HAPPEN!')
                        import pdb; pdb.set_trace()
                # Other parameters
                else:
                    ind, = np.where(fitnames==name)
                    labels[k] = args[ind[0]]

        # Add values for individual elements we are fitting (not FE_H or ALPHA_H)
        #   The code above misses individual elements that are being fit
        gdelem, = np.where(fitnames.endswith('_H') & (fitnames.find('FE_H')==-1) &
                           (fitnames.find('ALPHA_H')==-1))
        ngdelem = len(gdelem)
        for k in range(ngdelem):
            name = fitnames[gdelem[k]]
            ind, = np.where(labelnames==name)
            labels[ind[0]] = args[gdelem[k]]
        
        return labels
    
    def chisq(self,model):
        """
        Calculate the chi-squared between the Payne model spectrum and observed spectrum.
       
        Parameters
        ----------
        model : numpy array
           Array of Payne model spectrum flux values.

        Returns
        -------
        chisq : float
           Chi-squared value of the input Payne model spectrum and the observed spectrum.

        Example
        -------
        .. code-block:: python

             chisq = spfitter.chisq(model)

        """
        return np.sqrt( np.sum( (self._flux-model)**2/self._err**2 )/len(self._flux) )
            
    def model(self,xx,*args):
        """
        Return model Payne spectrum given the input arguments.  To be used with
        curve_fit().

        Parameters
        ----------
        xx : numpy array
            Input indepedent wavelength values.  Not used, but needed for curve_fit().
        args : tuple
            Tuple of input positional arguments of fitted model labels.

        Returns
        -------
        mflux : numpy array
            The output model Payne spectrum flux array, flattened.

        Example
        -------
        .. code-block:: python

             mflux = spfitter.model(wave,*labels)

        """        
        # Convert arguments to Payne model inputs
        labels = self.mklabels(args)
        if self.verbose: print(args)
        self.nfev += 1
        return self._paynemodel(labels).flux.flatten()  # only return the flattened flux

    def mkdxlim(self,fitparams):
        """
        Make array of parameter changes at which curve_fit should finish.

        Parameters
        ----------
        fitparams : list
            List of parameter names.

        Returns
        -------
        dx_lim : numpy array
            Array of parameter changes at which curve_fit should finish.

        Example
        -------
        .. code-block:: python

             dx_lim = spfitter.mkdxlim(fitparams)

        """
        return mkdxlim(fitparams)

    def mkbounds(self,labels,initpars=None):
        """
        Make upper and lower bounds for Payne labels.
        
        Parameters
        ----------
        labels : list
            List of parameter names.
        initpars : numpy array, optional
            Input list of initial label guesses.  Optional

        Returns
        -------
        bounds : tuple
            Two-element tuple of lower and upper boundaries for the input labels.

        Example
        -------
        .. code-block:: python
        
             bounds = spfitter.mkbounds(labels,initpars)

        """
        return mkbounds(labels,initpars=initpars)

    def getstep(self,name,val=None,relstep=0.02):
        """
        Calculate step for a single parameter to be used to generate the Jacobian.
        """
        if name=='TEFF':
            step = 5.0
        elif name=='RV':
            step = 0.1
        elif (name=='VROT' or name=='VSINI'):
            step = 1.0
        elif (name=='VMICRO' or name=='VTURB'):
            step = 0.25
        elif name=='VMACRO':
            step = 0.25
        elif name.endswith('_H'):
            step = 0.01
        else:
            step = 0.01
        return step
                
    def jac(self,x,*args):
        """
        Compute the Jacobian matrix (an Npix-by-Npar matrix, where element (i, j)
        is the partial derivative of f[i] with respect to x[j]).  This is to be
        used with curve_fit().

        Parameters
        ----------
        args : tuple
            Tuple of input positional arguments of fitted model labels at which
            to calculate the Jacobian.

        Returns
        -------
        jac : numpy array
           Jacobian matrix (an Npix-by-Npar matrix) of how the model changes
           (at each pixel) with respect to each parameter.

        Example
        -------
         .. code-block:: python
              
              jac = spfitter.jac(wave,*args)

        """
        
        # Boundaries
        lbounds,ubounds = self.mkbounds(self.fitparams)
        
        relstep = 0.02
        npix = len(x)
        npar = len(args)
        
        # Create synthetic spectrum at current values
        f0 = self.model(self._wave,*args)
        #self.nfev += 1
        
        # Save models/pars/chisq
        self._all_pars.append(list(args).copy())
        self._all_model.append(f0.copy())
        self._all_chisq.append(self.chisq(f0))
        chisq = np.sqrt( np.sum( (self._flux-f0)**2/self._err**2 )/len(self._flux) )
        #if self.verbose:
        #    print('chisq = '+str(chisq))
        
        # Initialize jacobian matrix
        jac = np.zeros((npix,npar),np.float64)
        
        # Loop over parameters
        for i in range(npar):
            pars = np.array(copy.deepcopy(args))
            step = self.getstep(self.fitparams[i],pars[i],relstep)
            # Check boundaries, if above upper boundary
            #   go the opposite way
            if pars[i]>ubounds[i]:
                step *= -1
            pars[i] += step
            
            #if self.verbose:
            #    print('--- '+str(i+1)+' '+self.fitparams[i]+' '+str(pars[i])+' ---')

            f1 = self.model(self._wave,*pars)
            #self.nfev += 1
            
            # Save models/pars/chisq
            self._all_pars.append(list(pars).copy())
            self._all_model.append(f1.copy())
            self._all_chisq.append(self.chisq(f1))
            
            if np.sum(~np.isfinite(f1))>0:
                print('some nans/infs')
                import pdb; pdb.set_trace()

            jac[:,i] = (f1-f0)/step
            
        if np.sum(~np.isfinite(jac))>0:
            print('some nans/infs')
            import pdb; pdb.set_trace()

        self._jac_array = jac.copy()   # keep a copy
            
        self.njac += 1
            
        return jac
