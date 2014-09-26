# quicklens/cinv/__init__.py
# --
# this module contains routines for inverse-variance
# filtering of flat-sky CMB maps.
# the submodules are:
#
# + cmb specific libraries
#  * opfilt_t    = operations and filters for inverse-variance filtering of a temperature-only map.
#  * opfilt_teb  = operations and filters for inverse-variance filtering of a temperature+polarization map.
#  * multigrid   = tools for performing inverse-variance filtering on a cmb map using multigrid-preconditioned gradient descent (following arxiv:0705.3980).
#
# + general tools for matrix inversion using conjugate descent.
#  * cg_solve    = solvers for multiplication by a matrix inverse using conjugate gradients.
#  * cd_solve    = solvers for multiplication by a matrix inverse using conjucate directions.
#  * cd_monitors = monitoring utilities for conjugate descent.
#

#import opfilt_t
#import opfilt_teb
#import multigrid

import cd_monitors
import cd_solve
import cg_solve
