# quicklens/lens.py
# --
# this module contains subroutines for applying the lensing operation to flat-sky maps.

import numpy as np
import scipy.interpolate
import scipy.misc

import maps
import qest

def calc_lensing_b_first_order( e, phi ):
    """ evaluate the lensed B-mode to first-order in phi on the flat-sky.
         * e   = unlensed E-modes (either a maps.rmap, maps.rfft, or maps.cfft object).
         * phi = cmb lensing potential (either a maps.rmap, maps.rfft, or maps.cfft object).

        returns a maps.cfft object containing the lensed B-modes to first-order in phi.
    """
    if maps.is_rmap(e):
        e = e.get_rfft().get_cfft()
    if maps.is_rfft(e):
        e = e.get_cfft()

    if maps.is_rmap(phi):
        pfft = phi.get_rfft().get_cfft()
    elif maps.is_rfft(phi):
        pfft = phi.get_cfft()
    else:
        pfft = phi
        
    assert( e.compatible(pfft) )

    ret = maps.cfft(nx=e.nx, dx=e.dx, ny=e.ny, dy=e.dy)

    lx, ly  = ret.get_lxly()
    l       = np.sqrt(lx**2 + ly**2)
    psi     = np.arctan2(lx, -ly)

    ret.fft = -0.5j * ( np.fft.fft2( +np.fft.ifft2(lx*e.fft*np.exp(+2.j*psi)) * np.fft.ifft2(lx*pfft.fft) +
                                     +np.fft.ifft2(ly*e.fft*np.exp(+2.j*psi)) * np.fft.ifft2(ly*pfft.fft) ) * np.exp(-2.j*psi) +
                        np.fft.fft2( -np.fft.ifft2(lx*e.fft*np.exp(-2.j*psi)) * np.fft.ifft2(lx*pfft.fft) +
                                     -np.fft.ifft2(ly*e.fft*np.exp(-2.j*psi)) * np.fft.ifft2(ly*pfft.fft) ) * np.exp(+2.j*psi) )
    ret    *= np.sqrt(ret.nx * ret.ny) / np.sqrt(ret.dx * ret.dy)
    return ret

def calc_lensing_clbb_flat_sky_first_order(lbins, nx, dx, cl_unl, t=None):
    """ evaluate the lensed B-mode power spectrum at first order in |phi|^2 on the flat-sky.
         * lbins  = array or list containing boundaries for the binned return spectrum.
         * nx     = width of the grid (in pixels) used to peform the calculation.
         * dx     = width of each pixel (in radians) used to perform the calculation.
         * cl_unl = object with attributes
                      .lmax = maximum multipole
                      .clee = unlensed E-mode power spectrum
                      .clpp = lensing potential power spectrum
         * (optional) w = weight function w(l) to be used when binning the return spectrum.

         returns a spec.bcl object containing the binned clbb power spectrum.
    """
    ret = maps.cfft(nx, dx)
    qeep = qest.lens.blen_EP( np.sqrt(cl_unl.clee), np.sqrt(cl_unl.clpp) )
    qeep.fill_resp( qeep, ret, np.ones(cl_unl.lmax+1), 2.*np.ones(cl_unl.lmax+1), npad=1 )
    return ret.get_ml(lbins, t=t)

def calc_lensing_clbb_flat_sky_first_order_curl(lbins, nx, dx, cl_unl, t=None):
    """ version of calc_lensing_clbb_flat_sky_first_order which treats cl_unl as a curl-mode lensing potential psi rather than as a gradient mode phi. """
    ret = maps.cfft(nx, dx)
    qeep = qest.qest_blen_EX( np.sqrt(cl_unl.clee), np.sqrt(cl_unl.clpp) )
    qeep.fill_resp( qeep, ret, np.ones(cl_unl.lmax+1), 2.*np.ones(cl_unl.lmax+1) )
    return ret.get_ml(lbins, t=t)

def make_lensed_map_flat_sky( tqumap, phifft, psi=0.0 ):
    """ perform the remapping operation of lensing in the flat-sky approximation.
         tqumap         = unlensed tqumap object to sample from.
         phifft         = phi field to calculate the deflection d=\grad\phi from.
         (optional) psi = angle to rotate the deflection field by, in radians (e.g. psi=pi/2 results in phi being treated as a curl potential).
    """
    assert( maps.pix.compatible( tqumap, phifft ) )

    lx, ly = phifft.get_lxly()
    nx, ny = phifft.nx, phifft.ny
    dx, dy = phifft.dx, phifft.dy

    pfft   = phifft.fft

    # deflection field
    x, y   = np.meshgrid( np.arange(0,nx)*dx, np.arange(0,ny)*dy )
    gpx    = np.fft.irfft2( pfft * lx * -1.j * np.sqrt( (nx*ny)/(dx*dy) ) )
    gpy    = np.fft.irfft2( pfft * ly * -1.j * np.sqrt( (nx*ny)/(dx*dy) ) )

    if psi != 0.0:
        gp = (gpx + 1.j*gpy)*np.exp(1.j*psi)
        gpx = gp.real
        gpy = gp.imag

    lxs    = (x+gpx).flatten(); del x, gpx
    lys    = (y+gpy).flatten(); del y, gpy

    # interpolate
    tmap   = scipy.interpolate.RectBivariateSpline( np.arange(0,ny)*dy, np.arange(0,nx)*dx, tqumap.tmap ).ev(lys, lxs).reshape([ny,nx])
    qmap   = scipy.interpolate.RectBivariateSpline( np.arange(0,ny)*dy, np.arange(0,nx)*dx, tqumap.qmap ).ev(lys, lxs).reshape([ny,nx])
    umap   = scipy.interpolate.RectBivariateSpline( np.arange(0,ny)*dy, np.arange(0,nx)*dx, tqumap.umap ).ev(lys, lxs).reshape([ny,nx])

    return maps.tqumap( nx, dx, [tmap, qmap, umap], ny = ny, dy = dy )
