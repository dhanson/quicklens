#/usr/bin/env python
# --
# quicklens/examples/lens/make_lensed_map_flatsky.py
# --
# generates a set of lensed maps in the flat-sky
# approximation and then compares their TT, EE, and BB
# power spectra to the expectation from CAMB.

import numpy as np
import pylab as pl

import quicklens as ql

# set simulation parameters.
nsims      = 10
lmax       = 3500
nx         = 256
dx         = 2./60./180.*np.pi
t          = lambda l:l*(l+1.)/(2.*np.pi)

lbins      = np.linspace(2, lmax, 100)

cl_unl     = ql.spec.get_camb_scalcl(lmax=lmax)
cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)

pix        = ql.maps.pix(nx, dx)

# run simulations.
bl_unl_avg = ql.util.avg()
bl_len_avg = ql.util.avg()
for idx, i in ql.util.enumerate_progress(np.arange(0, nsims)):
    teb_unl  = ql.sims.tebfft(pix, cl_unl)
    tqu_unl  = teb_unl.get_tqu()
    teb_unl  = tqu_unl.get_teb()

    phifft   = ql.sims.rfft( pix, cl_unl.clpp )
    tqu_len  = ql.lens.make_lensed_map_flat_sky( tqu_unl, phifft )
    teb_len  = tqu_len.get_teb()

    bl_unl_avg.add( teb_unl.get_cl( lbins, t=t ) )
    bl_len_avg.add( teb_len.get_cl( lbins, t=t ) )  

bl_unl = bl_unl_avg.get()
bl_len = bl_len_avg.get()

# make plots
pl.figure()

cl_unl.plot('cltt', t=t, color='k', label='Unlensed Theory' )
cl_len.plot('cltt', t=t, color='m', label='Lensed Theory')
pl.loglog( bl_unl.ls, bl_unl.cltt, color='y', label='Unlensed Sims' )
pl.loglog( bl_len.ls, bl_len.cltt, color='b', label='Lensed Sims' )

cl_unl.plot('clee', t=t, color='k')
cl_len.plot('clee', t=t, color='m')
pl.loglog( bl_unl.ls, bl_unl.clee, color='y' )
pl.loglog( bl_len.ls, bl_len.clee, color='b' )

cl_len.plot('clbb', t=t, color='m')
pl.loglog( bl_len.ls, bl_len.clbb, color='b' )

pl.xlabel(r'$l$')
pl.ylabel(r'$l(l+1)C_l$')

pl.legend(loc='lower right')
pl.setp( pl.gca().get_legend().get_frame(), visible=False )

pl.ion()
pl.show()
