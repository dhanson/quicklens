# quicklens/cinv/__init__.py
# --
# this module contains routines for C^{-1}  filtering of flat-sky CMB maps.
# the submodules are:
#
# + cmb specific libraries
#  * opfilt_teb  = operations and filters for C^{-1} filtering of a temperature+polarization map.
#  * multigrid   = tools for performing C^{-1} filtering on a cmb map using multigrid-preconditioned gradient descent (following arxiv:0705.3980).
#
# + general tools for matrix inversion using conjugate descent.
#  * cg_solve    = solvers for multiplication by a matrix inverse using conjugate gradients.
#  * cd_solve    = solvers for multiplication by a matrix inverse using conjucate directions.
#  * cd_monitors = monitoring utilities for conjugate descent.
#

import opfilt_teb
import multigrid

import cd_monitors
import cd_solve
import cg_solve
