# quicklens/math/wignerd.py
# --
# routines for wigner d-matrix sums and
# integrals using Gauss-Legendre quadrature.

import numpy as np
import cwignerd

class gauss_legendre_quadrature:
    """
    this is a class for performing integrals and sums over
    Wigner D matrices using Gauss-Legendre quadrature.
    it is taken from libkms_ist, by Kendrick Smith.
    
      self.npoints = number of points in quadrature
      self.zvec    = list of points in [-1,1]
      self.wvec    = integration weights for each point
    """

    def __init__(self, npoints):
        """ initialize the quadrature, which will provide a basis for performing exact integrations of polynomials with order up to some integer 'npoints'. """
        self.npoints = npoints
        self.zvec, self.wvec = cwignerd.init_gauss_legendre_quadrature(npoints)

    def cf_from_cl(self, s1, s2, cl):
        """ this computes cf[j] = \sum_{l} cl[l] d^l_{s1 s2}(self.zvec[j]). """

        lmax = len(cl)-1

        if np.iscomplexobj(cl):
            #FIXME: convert to 1 cf_from_cl call for potential 2x speed boost.
            return (cwignerd.wignerd_cf_from_cl( s1, s2, 1, self.npoints, lmax, self.zvec, cl.real ) +
                    cwignerd.wignerd_cf_from_cl( s1, s2, 1, self.npoints, lmax, self.zvec, cl.imag ) * 1.j) 
        else:
            return (cwignerd.wignerd_cf_from_cl( s1, s2, 1, self.npoints, lmax, self.zvec, cl ))

    def cl_from_cf(self, lmax, s1, s2, cf):
        """ this computes cl[l] = \int_{x} cf(x) d^l_{s1 s2}(x)
                               \sum_{j} cf[j] d^l_{s1 s2}[self.zvec[j]] * self.wvec[j]
                               for an array cf[j] which represents a polynomial cf(x)
                               with degree < self.npoints - lmax sampled at x=self.zvec[j]. """

        if np.iscomplexobj(cf):
            #FIXME: convert to 1 cl_from_cf call for potential 2x speed boost.
            return (cwignerd.wignerd_cl_from_cf( s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf.real ) +
                    cwignerd.wignerd_cl_from_cf( s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf.imag ) * 1.j)
        else:
            return (cwignerd.wignerd_cl_from_cf( s1, s2, 1, self.npoints, lmax, self.zvec, self.wvec, cf ))
