from distutils.core import setup
from Cython.Build import cythonize

setup(name='Uniform n-ball point picking.',
      ext_modules=cythonize("ballpoint.pyx"))
