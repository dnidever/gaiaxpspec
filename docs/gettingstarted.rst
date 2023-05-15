***************
Getting Started
***************



Introduction
============

This is a Python package to work with Gaia XPSpec data.  You can determine the stellar parameters using
an APOGEE empirical library or using an artificial neural network (ANN) model with added extinction.

Getting stellar parameters using the APOGEE empirical library
=============================================================

.. code-block:: python
                
    from gaiaxpspec import utils,fitter

    # Load the library spectra
    library = Table.read('apogee_library_spectra.fits')    
    # Determine the best-matching stellar parameter for the "flux" spectrum
    out = fitter.library_estimate(flux,library=library)


Using the ANN Model
===================


.. code-block:: python

    from gaiaxpspec import utils,fitter,model,xpspec
    # Load the model
    xpm = model.load_model()
    # Make XPSpec object
    spec = xpspec.XPSpec(flux,err=flux_error)
    spec.normalize()   # normalize
    # Make model spectrum
    mspec = xpm([teff,logg,feh,0.0])

    # Run the fitter
    out,bestspec = fitter.solvestar(spec,initpar=initpar,bounds=bounds)
