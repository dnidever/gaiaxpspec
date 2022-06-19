#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='gaiaxpspec',
      version='1.0.0',
      description='Fit Gaia Bp/Rp spectra with models',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/gaiaxpspec',
      packages=['gaiaxpspec'],
      package_dir={'':'python'},
      scripts=['bin/gaiaxpspec'],
      install_requires=['numpy','astropy(>=4.0)','scipy','dlnpyutils(>=1.0.3)','gaiaxpy','dill','emcee'],
      include_package_data=True
)
