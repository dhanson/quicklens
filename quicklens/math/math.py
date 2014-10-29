# quicklens/math/math.py
# --
# this module contains routines for doing geometry on the sphere
# and template projection. overview of functions:
#     * radec_to_thetaphi = convert ra, dec (in degrees) to \theta, \phi (in radians).
#     * cross             = return the cross-product of two vectors.
#     * ncross            = return the unit-normalized cross-product of two vectors.
#     * erot_zyz          = perform a ZYZ Euler rotation.
#     * xyz2tp            = convert a unit-length x,y,z vector to \theta, \phi.
#     * tp2xyz            = convert vectors of \theta, \phi to vectors of x, y, z on the unit sphere.
#     * proj_cof          = find projection coefficients for a set of templates to a data vector.
#     * proj_dat          = applies the coefficients found by proj_cof to remove the template vectors from a data vector.

import numpy as np

def radec_to_thetaphi(ras, decs):
    """ convert vectors of right ascension (ra) and declination (dec) both degrees to theta, phi in radians. """
    thts    = np.pi/2. - np.deg2rad(decs)  # radians. 0 at north pole up to +pi at south pole
    phis    = np.deg2rad(ras)  # for continuous (unwrapped) ra, this may go negative or above 2pi.
    return (thts, phis)

def cross( v1x, v1y, v1z, v2x, v2y, v2z ):
    """ helper function to return the cross-product of the vectors v1 and v2. """

    rx = v1y * v2z - v2y * v1z
    ry = v1z * v2x - v2z * v1x
    rz = v1x * v2y - v2x * v1y
    return rx, ry, rz

def ncross( v1x, v1y, v1z, v2x, v2y, v2z ):
    """ helper function to return the unit-normalized cross-product of the vectors v1 and v2. """
    
    rx = v1y * v2z - v2y * v1z
    ry = v1z * v2x - v2z * v1x
    rz = v1x * v2y - v2x * v1y
    rm = np.sqrt( rx**2 + ry**2 + rz**2 )
    return rx/rm, ry/rm, rz/rm

def erot_zyz( xs, ys, zs, psi, tht, phi ):
    """ returns the result of a ZYZ euler
    rotation on the vector(s) [xs, ys, zs].
    this is given by 3 consecutive right-
    handed rotations:
        1) about z by psi.
        2) about y by tht.
        3) about z by phi.
    """

    #rotate about z by psi.
    x1 = np.cos(psi) * xs - np.sin(psi) * ys
    y1 = np.sin(psi) * xs + np.cos(psi) * ys

    #rotate about y by tht.
    z2 = np.cos(tht) * zs - np.sin(tht) * x1
    x2 = np.sin(tht) * zs + np.cos(tht) * x1

    #rotate about z by phi.
    x3 = np.cos(phi) * x2 - np.sin(phi) * y1
    y3 = np.sin(phi) * x2 + np.cos(phi) * y1

    return x3, y3, z2

def xyz2tp(xs, ys, zs, center_phi=np.pi):
    """ converts a unit-length x,y,z vector to \theta, \phi.
    Put 2pi discontinuity opposite from center_phi (in radians).
    Default returns phi in [0,2pi). """
    ts = np.arccos(zs)
    ps = np.arctan2(ys, xs)
    return ts, ps

def tp2xyz(thetas, phis):
    """ converts vectors of \theta, \phi to unit vectors xs, ys, zs. """
    xs = np.sin(thetas) * np.cos(phis)
    ys = np.sin(thetas) * np.sin(phis)
    zs = np.cos(thetas)

    return xs, ys, zs

def proj_cof( d, vs, w=None ):
    """ given a data vector (d), a set of template vectors (vs), and an optional weight vector (w), 
    return a vector of coefficients (c) such that sum( (w*(d - c * vs))^2 ) is minimized.
    """
    if w == None:
        w = np.ones(len(y))

    n = len(vs)
    ptnd = np.zeros(n) 
    ptnp = np.zeros((n,n))
    for i in xrange(0, n):
        ptnd[i] = np.dot( vs[i], d*w )
        for j in xrange(0, n):
            ptnp[i,j] = np.dot( vs[i], vs[j] * w )

    c = np.dot( np.linalg.pinv(ptnp), ptnd )
    return c

def proj_dat( d, vs, w=None ):
    """ projects the templates (vs) out of d. see proj_cof for more details. """
    c = proj_cof( d, vs, w=w )
    
    r = np.copy(d)
    for i in xrange(0,n):
        r -= vs[i] * c[i]
    return r

def pad_ft(a, npad=2):
    """ pad a 2D Fourier transform (produced by np.fft2) with zeros, useful when performing convolutions to ensure that the result is not aliased.
           * a    = 2D complex Fourier transform.
           * npad = fractional size to pad. npad=2 will double the size of the Fourier transform in each dimension.
    """
    if npad==1: return a
    nx,ny = a.shape
    p = np.zeros([nx*npad,ny*npad], dtype=a.dtype)
    p[0:nx,0:ny] = np.fft.fftshift(a)
    p = np.roll(np.roll(p,-nx/2,axis=0),-ny/2,axis=1)
    return p

def unpad_ft(a, npad=2):
    """ un-pad an array in Fourier-space, removing the additional zeros added by 'pad_ft' """
    if npad==1: return a
    nx_pad,ny_pad = a.shape
    nx = int(nx_pad/npad); ny=int(ny_pad/npad)
    return np.roll(np.roll(
            (np.roll(np.roll(a,nx/2,axis=0),ny/2,axis=1)[0:nx,0:ny]),
            nx/2,axis=0),ny/2,axis=1)

def convolve_padded(f, g, npad=2):
    """ convolve two 2D complex Fourier transforms 'f' and 'g', using padding by a factor of npad to avoid aliasing.
           returns r(L) = \int{d^2\vec{l}} f(l) g(L-l).
    """
    return (unpad_ft(np.fft.fft2(
                np.fft.ifft2(pad_ft(f,npad=npad)) *
                np.fft.ifft2(pad_ft(g,npad=npad))),
                     npad=npad)*npad**2)
