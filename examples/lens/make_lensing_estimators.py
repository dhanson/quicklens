#/usr/bin/env python
# --
# quicklens/examples/lens/make_lensing_estimators.py
# --
# generates a set of lensed maps in the flat-sky limit, then runs
# quadratic lensing estimators on them to estimate phi. plots the
# auto-spectrum of the phi estimates as well as their cross-spectra
# with the input phi realization, and a semi-analytical estimate
# of the noise bias.

import numpy as np
import pylab as pl

import quicklens as ql

# simulation parameters.
nsims      = 25
lmin       = 100
lmax       = 3000
nx         = 256 # number of pixels.
dx         = 1./60./180.*np.pi # pixel width in radians.

nlev_t     = 5.  # temperature noise level, in uK.arcmin.
nlev_p     = 5.  # polarization noise level, in uK.arcmin.
bl         = ql.spec.bl(1., lmax) # beam transfer function.

pix        = ql.maps.pix(nx,dx)

# analysis parameters
estimators = [ ('ptt', 'r'), ('pee', 'g'), ('peb', 'b') ] # (estimator, plotting color color) pairs ('ptt' = TT estimator, etc.)

mask       = np.ones( (nx, nx) ) # mask to apply when inverse-variance filtering.
                                 # currently, no masking.
                                 # alternatively, cosine masking:
                                 # x, y = np.meshgrid( np.arange(0,nx), np.arange(0,nx) )
                                 # mask = np.sin( np.pi/nx*x )*np.sin( np.pi/nx*y )

mc_sims_mf = None                # indices of simulations to use for estimating a mean-field.
                                 # currently, no mean-field subtraction.
                                 # alternatively: np.arange(nsims, 2*nsims)

npad       = 1

# plotting parameters.
t          = lambda l: (l+0.5)**4/(2.*np.pi) # scaling to apply to cl_phiphi when plotting.
lbins      = np.linspace(10, lmax, 20)       # multipole bins.

# cosmology parameters.
cl_unl     = ql.spec.get_camb_scalcl(lmax=lmax)
cl_len     = ql.spec.get_camb_lensedcl(lmax=lmax)
clpp       = ql.spec.cl2cfft(cl_unl.clpp, ql.maps.cfft(nx,dx)).get_ml(lbins, t=t)

# make libraries for simulated skies.
sky_lib    = ql.sims.cmb.library_flat_lensed(pix, cl_unl, "temp/sky")
obs_lib    = ql.sims.obs.library_white_noise(pix, bl, sky_lib, nlev_t=nlev_t, nlev_p=nlev_p, lib_dir="temp/obs")

# make libraries for inverse-variance filtered skies.
cl         = ql.spec.clmat_teb( ql.util.dictobj( {'lmax' : lmax, 'cltt' : cl_len.cltt, 'clee' : cl_len.clee, 'clbb' : cl_len.clbb} ) )
transf     = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : lmax, 'cltt' : bl, 'clee' : bl, 'clbb' : bl} ), pix)
ivf_lib    = ql.sims.ivf.library_l_mask( ql.sims.ivf.library_diag(obs_lib, cl=cl, transf=transf, nlev_t=nlev_t, nlev_p=nlev_p, mask=mask), lmin=lmin, lmax=lmax )

qest_lib = ql.sims.qest.library(cl_unl, cl_len, ivf_lib, lib_dir="temp/qest", npad=npad)
qest_lib_kappa = ql.sims.qest.library_kappa(qest_lib, sky_lib)

qecl_lib = ql.sims.qecl.library(qest_lib, lib_dir="temp/qecl", mc_sims_mf=mc_sims_mf, npad=npad)
qecl_kappa_cross_lib = ql.sims.qecl.library(qest_lib, qeB=qest_lib_kappa, lib_dir="temp/qecl_kappa", npad=npad)

# --
# run estimators, make plots.
# --

pl.figure()

p = pl.semilogy

cl_unl.plot('clpp', t=t, color='k', p=p)

for key, color in estimators:
    qr = qest_lib.get_qr(key)
    qcr = qecl_lib.get_qcr_lcl(key)

    # intialize averagers.
    cl_phi_x_phi_avg    = ql.util.avg()
    cl_phi_x_est_avg    = ql.util.avg()
    cl_est_x_est_avg    = ql.util.avg()
    cl_est_x_est_n0_avg = ql.util.avg()

    # average power spectrum estimates.
    for idx, i in ql.util.enumerate_progress(np.arange(0, nsims), label="averaging cls for key=%s"%key):
        cl_est_x_est_n0_avg.add( (qecl_lib.get_sim_ncl_lcl(key, i) / qcr).get_ml(lbins, t=t) )
        cl_est_x_est_avg.add( (qecl_lib.get_sim_qcl_lcl(key, i) / qcr).get_ml(lbins, t=t) )
        cl_phi_x_est_avg.add( (qecl_kappa_cross_lib.get_sim_qcl_lcl(key, i) / qcr).get_ml(lbins, t=t) )

    # plot lensing spectra.
    (1./qr).get_ml(lbins, t=t).plot(color='y')            # analytical n0 bias.
     
    cl_phi_x_est_avg.plot(p=p, color=color)               # lensing estimate x input lensing. 
    cl_est_x_est_avg.plot(p=p, color=color, lw=3)         # lensing estimate auto spectrum.
    cl_est_x_est_n0_avg.plot(p=p, color=color, ls='--')   # semi-analytical n0 bias.

    (clpp + cl_est_x_est_n0_avg).plot(color='k', ls=':')  # theory lensing power spectrum + semi-analytical n0 bias.

pl.xlabel(r'$l$')
pl.ylabel(r'$(l+\frac{1}{2})^4 C_l^{\phi\phi} / 2\pi$')

pl.xlim(lbins[0], lbins[-1])


for key, color in estimators:
    pl.plot( [-1,-2], [-1,-2], label=r'$\phi^{' + key[1:].upper() + r'}$', lw=2, color=color )
pl.legend(loc='lower left')
pl.setp(pl.gca().get_legend().get_frame(), visible=False)
    
pl.ion()
pl.show()
