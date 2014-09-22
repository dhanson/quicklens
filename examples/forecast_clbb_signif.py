#/usr/bin/env python
# --
# quicklens/examples/forecast_clbb_signif.py
# --
# make a plot containing the Fisher-forecasted
# detection significance for lensing-induced
# B-mode power spectrum ClBB, for a range of
# different instrumental noise levels and
# minimum mulitpoles.

import numpy as np
import pylab as pl

import quicklens as ql

# --
lmin            = 10
lmax            = 2000
noise_uK_arcmin = 10.
fwhm_arcmin     = 1.0
coverage        = 100 # deg^2
fsky            = coverage/41000.
# --

cl = ql.spec.get_camb_lensedcl(lmax=lmax)

def clbb_sig(lmin, lmax, noise_uK_arcmin, fwhm_arcmin, ignore_cv=False):
    clnn = ql.spec.nl(noise_uK_arcmin, fwhm_arcmin, lmax)

    scal = {True : 0.0, False : 1.0}[ignore_cv]

    return np.sqrt( np.sum( cl.clbb[lmin:lmax+1]**2 / (scal * cl.clbb[lmin:lmax+1] + clnn[lmin:lmax+1])**2 * (2.*np.arange(lmin, lmax+1) + 1.) * 0.5 ) )

def print_clbb_sig(lmin, lmax, noise_uK_arcmin, fwhm_arcmin, ignore_cv=False):
    print ( "lmin = %4d, lmax = %4d, noise = %5.2f, fwhm = %5.2f, sig = %5.2f"
            % (lmin, lmax, noise_uK_arcmin, fwhm_arcmin,
               clbb_sig(lmin, lmax, noise_uK_arcmin, fwhm_arcmin, ignore_cv=ignore_cv) ) )

pl.figure(figsize=(7,5))
ls = np.arange(11, 2000)
for noise_uK_arcmin in [5, 7, 10, 15]:
    sigs = []
    for l in ls:
        sigs.append( clbb_sig(l, l, noise_uK_arcmin, fwhm_arcmin, ignore_cv=False)**2 )

    pl.subplot(121)
    pl.plot(ls, sigs / np.max(sigs), label=(r"$" + str(noise_uK_arcmin) + r"\mu K'$"), lw=2)

    pl.subplot(122)
    pl.plot(ls, np.sqrt(fsky*np.cumsum(sigs[::-1])[::-1]), label=(r"$" + str(noise_uK_arcmin) + r"\mu K'$"), lw=2)

# lhs
pl.subplot(121)
pl.title(r"$(\partial \sigma^{2}_{BB} / \partial l) / {\rm Max}_l(\partial \sigma^{2}_{BB} / \partial l)$ ")
pl.xlabel(r"$l$")
pl.ylim(0, 1.2)

# rhs
pl.subplot(122)
pl.title(r"${\rm Cumulative}$ ${\rm S/N}$")
pl.xlabel(r"$l_{\rm min}$")
pl.ylim(0, 20.)
for y in [5., 10., 15.]:
    pl.axhline(y=y, ls='--', color='gray')

pl.text( 0.5, 0.9, r"$f_{\rm sky} = " + str(coverage) + r" {\rm deg}^2$", transform=pl.gca().transAxes )

pl.subplot(121); pl.legend(loc='upper right')
pl.setp( pl.gca().get_legend().get_frame(), visible=False )

pl.ion()
pl.show()
