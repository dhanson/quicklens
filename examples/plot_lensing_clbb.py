#/usr/bin/env python
# --
# quicklens/examples/plot_lensing_clbb.py
# --
# calculate the lensing B-mode power spectrum
# to first order in the flat-sky limit, and
# compare to the full result produced by CAMB.

import numpy as np
import pylab as pl
import quicklens as ql

lbins = np.arange(50, 2000, 50) # multipole bins for the flat-sky calculation
nx = 512                        # dimension of grid for the flat-sky calculation.
dx = np.pi/180./60.             # pixel width = 1 arcminute

cl_unl = ql.spec.get_camb_scalcl(lmax=4000)
cl_len = ql.spec.get_camb_lensedcl(lmax=4000)

# calculate lensed clbb
clbb_first_order = ql.lens.calc_lensing_clbb_flat_sky_first_order(lbins, nx, dx, cl_unl)

# make plots
cl_len.plot('clbb', color='k', label="CAMB lensing ClBB")
clbb_first_order.plot(color='r', ls='--', lw=2, label="First-order flat-sky lensing ClBB")

pl.legend(loc='upper right'); pl.setp( pl.gca().get_legend().get_frame(), visible=False )
pl.xlabel(r'$L$')
pl.ylabel(r'$C_L^{BB}$')

pl.xlim(0,2000)

# display plots
pl.ion()
pl.show()
