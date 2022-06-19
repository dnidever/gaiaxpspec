#!/usr/bin/env python

"""XPSPEC.PY - The spectrum class and methods

"""

from __future__ import print_function

__authors__ = 'David Nidever <dnidever@montana.edu>'
__version__ = '20220618'  # yyyymmdd                                                                                                                           

import numpy as np
import math
import warnings
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.io import fits
from astropy.table import Table
import thecannon as tc
from dlnpyutils import utils as dln, bindata
import copy
from . import utils
import dill as pickle
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

fitz = Table.read(utils.datadir()+'fitzpatrick2019_gaiabprp_extinction.fits')

# Continuum normalize using median
def continuum_median(spec):
    return np.nanmedian(spec.flux)

# Continuum normalize using central 11 pixels
def continuum_central(spec,npix=11):
    cen = len(spec.flux)//2
    hpix = npix//2
    lo = cen-hpix
    hi = cen+hpix+1
    return np.nanmedian(spec.flux[lo:hi])

# Combine multiple spectra
def combine(speclist,wave=None,sum=False):
    """
    This function combines/stacks multiple spectra.

    Parameters
    ----------
    speclist : list of Spec1D objects
         The list of spectra to combine.
    wave : array, optional
        The output wavelength array.  By default, the wavelength array of
        the first spectrum is used.
    sum : bool, optional
        Sum the spectra instead of averaging them.  The default is to average.

    Returns
    -------
    spec : Spec1D object
        The final combined/stacked spectrum.

    Examples
    --------
    .. code-block:: python

        spec = combine([spec1,spec2,spec3])

    """
    
    if type(speclist) is list:
        nspec = len(speclist)
        spec1 = speclist[0]
    else:
        nspec = 1
        spec1 = speclist

    # Only one spectrum input but no wavelength, just return the spectrum
    if (nspec==1) & (wave is None):
        return speclist
    
    # Final wavelength scale, if not input then use wavelengths of first spectrum
    if wave is None:
        wave = spec1.wave

    # How many orders in output wavelength
    if (wave.ndim==1):
        nwpix = len(wave)
        nworder = 1
    else:
        nwpix,nworder = wave.shape
            
    # Loop over the spectra
    flux = np.zeros((nwpix,nworder,nspec),float)
    err = np.zeros((nwpix,nworder,nspec),float)
    mask = np.zeros((nwpix,nworder,nspec),bool)
    for i in range(nspec):
        if (nspec==1):
            spec = speclist
        else:
            spec = speclist[i]
        # Interpolate onto final wavelength scale
        spec2 = spec.interp(wave)
        flux[:,:,i] = spec1.flux
        err[:,:,i] = spec1.err
        mask[:,:,i] = spec1.mask
    # Weighted average
    wt = 1.0 / np.maximum(err,0.0001)**2
    aflux = np.sum(flux*wt,axis=2)/ np.sum(wt,axis=2)
    # uncertainty in weighted mean is 1/sqrt(sum(wt))
    # https://physics.stackexchange.com/questions/15197/how-do-you-find-the-uncertainty-of-a-weighted-average
    aerr = 1.0 / np.sqrt(np.sum(wt,axis=2))
    # mask is set only if all mask values are set
    #  and combine
    amask = np.zeros((nwpix,nworder),bool)
    for i in range(nspec):
        amask = np.logical_and(amask,mask[:,:,i])

    # Sum or average?
    if sum is True:
        aflux *= nspec
        aerr *= nspec

    # Create output spectrum object
    ospec = XPSpec(aflux,err=aerr)
    # cannot handle LSF yet

    return ospec

def fitz_extinct(A55,R55=3.02):
    # Get extinction
    # Fitzpatrick+2019, eq.8
    # k(lambda-55)_R(55) = k(lambda-55)0 + s(lambda-55) * (R(55)-R(55)0)
    # where R(55)0 = 3.02
    klambda55 = fitz['klambda55'] + fitz['slambda55'] * (R55-3.02)
    # From Fitzpatrick+2019 eq. 19
    # A(lambda)/A(V) = k(lambda-V)/R(V) + 1
    # or using 55 instead of V
    # A(lambda)/A(55) = k(lambda-55)/R(55) + 1
    alambda = A55 * (klambda55/R55+1)
    # Convert to linear attenuation
    attenuation = 10**(alambda/-2.5)
    return alambda, attenuation

# Object for representing 1D spectra
class XPSpec:
    """
    A class to represent Gaia Xp/RP combined spectra.

    Parameters
    ----------
    flux : array
        Array of flux values.  Can 2D if there are multiple orders, e.g. [Npix,Norder].
        The order dimension must always be the last/trailing dimension.
    err : array, optional
        Array of uncertainties in the flux array.  Same shape as flux.
    wave : array, optional
        Array of wavelengths for the flux array.  Same shape as flux.
    mask : boolean array, optional
        Boolean bad pixel mask array (good=False, bad=True) for flux.  Same shape as flux.
    bitmask : array, optional
        Bit mask giving specific information for each flux pixel.  Same shape as flux.

    """

    
    # Initialize the object
    def __init__(self,flux,err=None,wave=None,mask=None,bitmask=None,head=None,
                 instrument='GaiaXP',filename=None,continuum_func=continuum_median):
        """ Initialize Spec1D object."""
        self.flux = flux
        self.err = err
        self.wave = wave
        self.mask = mask
        if mask is None:
            self.mask = np.zeros(flux.shape,bool)
        self.bitmask = bitmask
        self.head = head
        self.instrument = instrument
        self.filename = filename
        self.snr = None
        if self.err is not None:
            self.snr = np.nanmedian(self.flux[~self.mask])/np.nanmedian(self.err[~self.mask])
        self.normalized = False
        if flux.ndim==1:
            npix = len(flux)
            norder = 1
        else:
            npix,norder = flux.shape
        self.ndim = flux.ndim
        self.npix = npix
        self.norder = norder
        self.continuum_func = continuum_func
        self._cont = None
        return
    
    def __repr__(self):
        """ Print out the string representation of the Spec1D object."""
        s = repr(self.__class__)+"\n"
        if self.instrument is not None:
            s += self.instrument+" spectrum\n"
        if self.instrument is not None:
            s += "Instrument = "+self.instrument+"\n"
        if self.filename is not None:
            s += "File = "+self.filename+"\n"            
        if self.snr is not None:
            s += ("S/N = %7.2f" % self.snr)+"\n"
        if self.norder>1:
            s += 'Dimensions: ['+str(self.npix)+','+str(self.norder)+']\n'
        else:
            s += 'Dimensions: ['+str(self.npix)+']\n'                                                       
        s += "Flux = "+str(self.flux)+"\n"
        if self.err is not None:
            s += "Err = "+str(self.err)+"\n"
        if self.wave is not None:
            s += "Wave = "+str(self.wave)
        return s

    def getcont(self,**kwargs):
        """ Calculate the continuum value. """
        return self.continuum_func(self,**kwargs)
    
    @property
    def cont(self,**kwargs):
        """ Return the continuum."""
        if self._cont is None:
            cont = self.getcont(**kwargs)
            self._cont = cont
        return self._cont

    @cont.setter
    def cont(self,inpcont):
        """ Set the continuum."""
        self._cont = inpcont

    def extinct(self,a55,r55=3.02):
        """ Apply extinction to the spectrum."""
        flux = self._flux.copy()
        self.a55 = a55
        self.r55 = r55
        
        # Extinct it
        if a55 > 0:
            # Get the extinction            
            alambda,attenuation = fitz_extinct(a55,r55)
            # Now apply the extinction to the model
            self.flux = flux*attenuation
            # Normalize again
            self.normalized = False
            self.normalize()
        
    def wave2pix(self,w,extrapolate=True,order=0):
        """
        Convert wavelength values to pixels using the spectrum dispersion
        solution.

        Parameters
        ----------
        w : array
          Array of wavelength values to convert.
        extrapolate : bool, optional
           Extrapolate beyond the boundaries of the dispersion solution,
           if necessary.  True by default.
        order : int, optional
            The order to use if there are multiple orders.
            The default is 0.
              
        Returns
        -------
        x : array
          The array of converted pixels.

        Example
        -------
        .. code-block:: python

             x = spec.wave2pix(w)

        """
        
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.wave.ndim==2:
            # Order is always the second dimension
            return utils.w2p(self.wave[:,order],w,extrapolate=extrapolate)            
        else:
            return utils.w2p(self.wave,w,extrapolate=extrapolate)

        
    def pix2wave(self,x,extrapolate=True,order=0):
        """
        Convert pixel values to wavelengths using the spectrum dispersion
        solution.

        Parameters
        ----------
        x : array
          Array of pixel values to convert.
        extrapolate : bool, optional
           Extrapolate beyond the boundaries of the dispersion solution,
           if necessary.  True by default.
        order : int, optional
            The order to use if there are multiple orders.
            The default is 0.
              
        Returns
        -------
        w : array
          The array of converted wavelengths.

        Example
        -------
        .. code-block:: python

             w = spec.pix2wave(x)

        """
        
        if self.wave is None:
            raise Exception("No wavelength information")
        if self.wave.ndim==2:
             # Order is always the second dimension
            return utils.p2w(self.wave[:,order],x,extrapolate=extrapolate)
        else:
            return utils.p2w(self.wave,x,extrapolate=extrapolate)            

        
    def normalize(self,**kwargs):
        """
        Normalize the spectrum using a specified continuum function.

        Parameters
        ----------
        **kwargs : arguments that are passed to the continuum function.
              
        Returns
        -------
        The flux and err arrays will normalized (divided) by the continuum
        and the continuum saved in cont.  The normalized property is set to
        True.

        Example
        -------
        .. code-block:: python

             spec.normalize()

        """

        if self.normalized is True:
            return
        
        # Use the continuum_func to get the continuum
        cont = self.getcont()
        if self.err is not None:
            self.err /= cont
        self.flux /= cont
        self.cont = cont
        self.normalized = True
        return
        
    def interp(self,x=None,xtype='wave',order=None):
        """ Interpolate onto a new wavelength scale and/or shift by a velocity."""
        # if x is 2D and has multiple dimensions and the spectrum does as well
        # (with the same number of dimensions), and order=None, then it is assumed
        # that the input and output orders are "matched".
        
        # Check input xtype
        if (xtype.lower().find('wave')==-1) & (xtype.lower().find('pix')==-1):
            raise ValueError(xtype+' not supported.  Must be wave or pixel')

        # Convert pixels to wavelength
        if (xtype.lower().find('pix')>-1):
            wave = self.pix2wave(x,order=order)
        else:
            wave = x.copy()
        
        # How many orders in output wavelength
        if (wave.ndim==1):
            nwpix = len(wave)
            nworder = 1
        else:
            nwpix,nworder = wave.shape
        wave = wave.reshape(nwpix,nworder)  # make 2D for order indexing
            
        # Loop over orders in final wavelength
        oflux = np.zeros((nwpix,nworder),float)
        oerr = np.zeros((nwpix,nworder),float)
        omask = np.zeros((nwpix,nworder),bool)
        osigma = np.zeros((nwpix,nworder),float)        
        for i in range(nworder):
            # Interpolate onto the final wavelength scale
            wave1 = wave[:,i]
            wr1 = dln.minmax(wave1)
            
            # Make spectrum arrays 2D for order indexing, [Npix,Norder]
            swave = self.wave.reshape(self.npix,self.norder)
            sflux = self.flux.reshape(self.npix,self.norder)            
            serr = self.err.reshape(self.npix,self.norder)
            # convert mask to integer 0 or 1
            if hasattr(self,'mask'):
                smask = np.zeros((self.npix,self.norder),int)                
                smask_bool = self.err.reshape(self.npix,self.norder)            
                smask[smask_bool==True] = 1
            else:
                smask = np.zeros((self.npix,self.norder),int)

            # The orders are "matched", one input for one output order
            if (nworder==self.norder) & (order is None):
                swave1 = swave[:,i]                
                sflux1 = sflux[:,i]
                serr1 = serr[:,i]
                ssigma1 = self.lsf.sigma(order=i)
                smask1 = smask[:,i]
                # Some overlap
                if (np.min(swave1)<wr1[1]) & (np.max(swave1)>wr1[0]):
                    # Fix NaN pixels
                    bd,nbd = dln.where(np.isfinite(sflux1)==False) 
                    if nbd>0:
                        sflux1[bd] = 1.0
                        serr1[bd] = 1e30
                        smask1[bd] = 1
                    ind,nind = dln.where( (wave1>np.min(swave1)) & (wave1<np.max(swave1)) )
                    oflux[ind,i] = dln.interp(swave1,sflux1,wave1[ind],extrapolate=False,assume_sorted=False)
                    oerr[ind,i] = dln.interp(swave1,serr1,wave1[ind],extrapolate=False,assume_sorted=False,kind='linear')
                    osigma[ind,i] = dln.interp(swave1,ssigma1,wave1[ind],extrapolate=False,assume_sorted=False)
                    mask_interp = dln.interp(swave1,smask1,wave1[ind],extrapolate=False,assume_sorted=False)                    
                    mask_interp_bool = np.zeros(nind,bool)
                    mask_interp_bool[mask_interp>0.4] = True
                    omask[ind,i] = mask_interp_bool
                    
            # Loop over all spectrum orders
            else:
                # Loop over spectrum orders
                for j in range(self.norder):
                    swave1 = swave[:,j]
                    sflux1 = sflux[:,j]
                    serr1 = serr[:,j]
                    ssigma1 = self.lsf.sigma(order=j)                    
                    smask1 = smask[:,j]
                    # Some overlap
                    if (np.min(swave1)<wr1[1]) & (np.max(swave1)>wr1[0]):
                        ind,nind = dln.where( (wave1>np.min(swave1)) & (wave1<np.max(swave1)) )
                        oflux[ind,i] = dln.interp(swave1,sflux1,wave1[ind],extrapolate=False,assume_sorted=False)
                        oerr[ind,i] = dln.interp(swave1,serr1,wave1[ind],extrapolate=False,assume_sorted=False,kind='linear')
                        osigma[ind,i] = dln.interp(swave1,ssigma1,wave1[ind],extrapolate=False,assume_sorted=False)
                        mask_interp = dln.interp(swave1,smask1,wave1[ind],extrapolate=False,assume_sorted=False)                    
                        mask_interp_bool = np.zeros(nind,bool)
                        mask_interp_bool[mask_interp>0.4] = True
                        omask[ind,i] = mask_interp_bool
                    # Currently this does NOT deal with the overlap of multiple orders (e.g. averaging)
        # Flatten if 1D
        if (x.ndim==1):
            wave = wave.flatten()
            oflux = oflux.flatten()
            oerr = oerr.flatten()
            osigma = osigma.flatten()            
            omask = omask.flatten()
        ospec = XPSpec(oflux,wave=wave,err=oerr,mask=omask)

        return ospec
                

    def copy(self):
        """ Create a new copy."""
        if self.err is not None:
            newerr = self.err.copy()
        else:
            newerr = None
        if self.wave is not None:
            newwave = self.wave.copy()
        else:
            newwave = None
        if self.mask is not None:
            newmask = self.mask.copy()
        else:
            newmask = None
        new = XPSpec(self.flux.copy(),err=newerr,wave=newwave,mask=newmask,
                     instrument=self.instrument,filename=self.filename)
        for name, value in vars(self).items():
            if name not in ['flux','wave','err','mask','instrument','filename']:
                setattr(new,name,copy.deepcopy(value))           

        return new

    def write(self,outfile,overwrite=True):
        """ Write the spectrum to a FITS file."""

        hdu = fits.HDUList()
        # header
        hdu.append(fits.PrimaryHDU(header=self.head))
        hdu[0].header['COMMENT'] = 'HDU0: header'
        hdu[0].header['COMMENT'] = 'HDU1: flux'
        hdu[0].header['COMMENT'] = 'HDU2: flux error'
        hdu[0].header['COMMENT'] = 'HDU3: wavelength'
        hdu[0].header['COMMENT'] = 'HDU4: mask'
        hdu[0].header['SPECTYPE'] = 'XPSPEC'
        hdu[0].header['NORMLIZD'] = self.normalized
        hdu[0].header['NDIM'] = self.ndim
        hdu[0].header['NPIX'] = self.npix
        hdu[0].header['NORDER'] = self.norder
        hdu[0].header['SNR'] = self.snr
        # flux
        hdu.append(fits.ImageHDU(self.flux))
        hdu[1].header['BUNIT'] = 'Flux'
        # error
        hdu.append(fits.ImageHDU(self.err))
        hdu[2].header['BUNIT'] = 'Flux Error'        
        # wavelength
        hdu.append(fits.ImageHDU(self.wave))
        hdu[3].header['BUNIT'] = 'Wavelength (Ang)'
        # mask
        hdu.append(fits.ImageHDU(self.mask.astype(int)))
        hdu[4].header['BUNIT'] = 'Mask'
        hdu.writeto(outfile,overwrite=overwrite)
