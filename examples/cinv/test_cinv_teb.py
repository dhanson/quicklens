# quicklens/examples/test_cinv_teb.py
# --
# test script which generates a C^{-1} filtered temperature + polarization CMB map.
#

import numpy as np
import pylab as pl

import quicklens as ql

multigrid  = False                                 # use a multigrid preconditioner.
eps_min    = 1.e-4
lmax       = 3000                                  # maximum multipole.
nx         = 512                                   # size of map (in pixels).
dx         = 1./60./180.*np.pi                     # size of pixels (in radians).

nlev_t     = 10.                                   # temperature map noise level, in uK.arcmin.
nlev_p     = 10.                                   # polarization map noise level (Q, U), in uK.arcmin.
bl         = ql.spec.bl(fwhm_arcmin=1., lmax=lmax) # instrumental beam transfer function.

cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)  # cmb theory spectra.

mask = np.zeros( (nx, nx) )                        # test mask.
mask[ nx/4:3*nx/4, nx/4:3*nx/4 ] = 1.0             # (rectangular cutout)

nltt       = (nlev_t*np.pi/180./60.)**2 / bl**2
nlee       = (nlev_p*np.pi/180./60.)**2 / bl**2

# diagonal approximation of filter.
flt        = 1.0/(nltt[0:lmax+1] + cl_len.cltt[0:lmax+1]); flt[0:2] = 0.0
fle        = 1.0/(nlee[0:lmax+1] + cl_len.clee[0:lmax+1]); fle[0:2] = 0.0
flb        = 1.0/(nlee[0:lmax+1] + cl_len.clbb[0:lmax+1]); flb[0:2] = 0.0

pix        = ql.maps.pix(nx, dx)

# simulate input sky
teb_sky  = ql.sims.tebfft(pix, cl_len)
tqu_sky  = teb_sky.get_tqu()

# simulate observed sky
teb_obs  = ql.spec.blmat_teb(bl) * teb_sky
tqu_obs  = teb_obs.get_tqu() + ql.sims.tqumap_homog_noise( pix, nlev_t, nlev_p )

# diagonal filter
teb_filt_diag = ql.spec.blmat_teb(1./bl) * (tqu_obs * mask).get_teb()
teb_filt_diag = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : lmax, 'cltt' : flt, 'clee' : fle, 'clbb' : flb} ) ) * teb_filt_diag

# construct cinv filter
ninv_filt = ql.cinv.opfilt_teb.ninv_filt(ql.spec.blmat_teb(bl),
                                         ql.maps.tqumap(nx, dx,
                                                        [ (180.*60./np.pi*dx)**2 / nlev_t**2 * np.ones( (nx,nx) ) * mask,
                                                          (180.*60./np.pi*dx)**2 / nlev_p**2 * np.ones( (nx,nx) ) * mask,
                                                          (180.*60./np.pi*dx)**2 / nlev_p**2 * np.ones( (nx,nx) ) * mask ]))
sinv_filt = ql.cinv.opfilt_teb.sinv_filt(cl_len)

pre_op = ql.cinv.opfilt_teb.pre_op_diag( sinv_filt, ninv_filt )


# run solver
if multigrid == True:
    chain = ql.cinv.multigrid.chain( 0, ql.cinv.opfilt_teb, sinv_filt, ninv_filt, plogdepth=2, eps_min=eps_min )
    teb_filt_cinv = chain.solve( ql.maps.tebfft( nx, dx ), tqu_obs )
else:
    monitor = ql.cinv.cd_monitors.monitor_basic(ql.cinv.opfilt_teb.dot_op(), iter_max=np.inf, eps_min=eps_min)
    
    teb_filt_cinv = ql.maps.tebfft( nx, dx )
    ql.cinv.cd_solve.cd_solve( x = teb_filt_cinv,
                               b = ql.cinv.opfilt_teb.calc_prep(tqu_obs, sinv_filt, ninv_filt),
                               fwd_op = ql.cinv.opfilt_teb.fwd_op(sinv_filt, ninv_filt),
                               pre_ops = [pre_op], dot_op = ql.cinv.opfilt_teb.dot_op(),
                               criterion = monitor, tr=ql.cinv.cd_solve.tr_cg, cache=ql.cinv.cd_solve.cache_mem() )
    teb_filt_cinv = ql.cinv.opfilt_teb.calc_fini( teb_filt_cinv, sinv_filt, ninv_filt)

# -- plot wiener-filtered spectra.
pl.figure()
p = pl.loglog
t = lambda l : l*(l+1.)/(2.*np.pi)

lbins = np.linspace(10, lmax, 50) # multipole bins for plotting.

c = ql.spec.clmat_teb(cl_len)

# plot theory spectra.
cl_len.plot( 'cltt', p=p, t=t, color='k' )
cl_len.plot( 'clee', p=p, t=t, color='k' )
cl_len.plot( 'clbb', p=p, t=t, color='k' )

# plot wiener-filtered spectra, as well as diagonal approximation.
t = lambda l : l*(l+1.)/(2.*np.pi)
(c * teb_filt_cinv).get_cl(lbins, t=t).plot( 'cltt', p=p, color='r', label='TT' )
(c * teb_filt_diag).get_cl(lbins, t=t).plot( 'cltt', p=p, color='gray', ls='--' )

(c * teb_filt_cinv).get_cl(lbins, t=t).plot( 'clee', p=p, color='m', label='EE' )
(c * teb_filt_diag).get_cl(lbins, t=t).plot( 'clee', p=p, color='gray', ls='--' )

(c * teb_filt_cinv).get_cl(lbins, t=t).plot( 'clbb', p=p, color='g', label='BB' )
(c * teb_filt_diag).get_cl(lbins, t=t).plot( 'clbb', p=p, color='gray', ls='--' )

# add axis details and curve labels.
pl.legend(loc='lower left')
pl.setp( pl.gca().get_legend().get_frame(), color='w')
pl.xlim(np.pi/(nx*dx),lmax*1.1)
pl.ylim(1.e-6, 1.e4)

pl.xlabel(r'$l$')
pl.ylabel(r'$l(l+1)C_l / 2\pi$')

pl.ion()
pl.show()
