#!/usr/bin/env python
# to compile fortran code for spherical harmonic transforms inplace, run
# python setup.py build_ext --inplace --fcompiler=gnu95

import os,glob
import numpy as np
import distutils

def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('quicklens',parent_package,top_path)

    config.add_extension('math.cwignerd', ['quicklens/math/wignerd.pyf', 'quicklens/math/wignerd.c'])
    if distutils.version.StrictVersion(np.version.version) > distutils.version.StrictVersion('1.6.1'):
        config.add_extension('shts.fsht', ['quicklens/shts/shts.f95'],
                             libraries=['gomp'], f2py_options=[],
                             extra_f90_compile_args=['-ffixed-line-length-1000', '-fopenmp'],
                             extra_compile_args=['-fopenmp'], extra_link_args=[],)
    else:
        config.add_extension('shts.fsht', ['quicklens/shts/shts.f95'],
                             libraries=['gomp'], f2py_options=[],
                             extra_compile_args=['-fopenmp'], extra_link_args=[],)

    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup

    packages = ["quicklens","quicklens.cinv","quicklens.qest","quicklens.sims","quicklens.shts","quicklens.math"]

    data_dir = os.path.join("data","cl","planck_wp_highL","*")
    package_data = { "quicklens" : [data_dir] }

    setup(packages=packages,
          package_data=package_data,
          configuration=configuration)
