"""
Use Cython to build the C extension and compile it into a Python module.
"""
from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(ext_modules=cythonize("_tesseroid.pyx"),
      include_dirs=[numpy.get_include()])
