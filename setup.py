

#https://docs.python.org/2/distutils/setupscript.html

from setuptools import setup
from distutils.core import Extension
import sys
import sysconfig
import os, os.path
import subprocess
import glob

# just in case ...
#https://github.com/blog/2104-working-with-submodules
    
#accumulate C extension
accumulate_c_src= glob.glob('exptool/basis/accumulate_c_ext/*.c')


accumulate_include_dirs= ['exptool/basis/accumulate_c_ext']

accumulate_c= Extension('exptool_accumulate_c',
                             sources=accumulate_c_src,
                             #libraries=pot_libraries,
                             include_dirs=accumulate_include_dirs)
                             #extra_compile_args=extra_compile_args,
                             #extra_link_args=extra_link_args)

ext_modules = []
ext_modules.append(accumulate_c)
                             
    
setup(name='exptool',
      version='0.1',
      description='EXP analysis in Python',
      author='Michael Petersen',
      author_email='mpete0@astro.umass.edu',
      license='New BSD',
      #long_description=long_description,
      url='http://github.com/michael-petersen/exptool',
      package_dir = {'galpy/': ''},
      packages=['exptool','exptool/analysis','exptool/basis',
                'exptool/io','exptool/orbits','exptool/utils'],
      #package_data={'galpy/df_src':['data/*.sav'],
      #              "": ["README.rst","README.dev","LICENSE","AUTHORS.rst"]},
      include_package_data=True,
      install_requires=['numpy>=1.7','scipy','matplotlib'],
      ext_modules=ext_modules,
      )


