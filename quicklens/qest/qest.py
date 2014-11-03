# quicklens/qest/qest.py
# --
# this module contains classes and routines for applying quadratic
# anisotropy estimators to CMB maps.
#
# these estimators are motivated as follows: first, we start with
# a source of statistical anisotropy 's', which is a 2D field having
# a Harmonic transform s(L). 's' induces couplings between modes of the
# CMB temperature and polarization observables X, Y \in {T, E, B} as
#
# < X(l_X) Y(l_Y) > = \int{d^2 L} W^{s, XY}(l_X, l_Y, L) s(L),
#
# where <> represents an ensemble average over realizations of the CMB
# with s(L) fixed and W^{s, XY} is a weight function associated with the
# A-B coupling induced by 's'. we can try to measure s(L) using a
# quadratic estimator q, defined by
#
# q^{XY}(L) = 1/2 \int{d^2 l_X} \int{d^2 l_Y}
#                    W^{s, XY}(l_X, l_Y, L) \bar{X}(l_X) \bar{Y}(l_Y)
#
# where \bar{X} and \bar{Y} are filtered observations of the X and Y
# fields. nominally, the cost of evaluating q(L) is O(lmax^{6}) (because
# l_X, l_Y, and L are each 2D fields with lmax^2 elements), however if
# the weight function can be written as a sum of separable terms in
# l_X, l_Y, and L then this can be done much faster. for a flat-sky map
# in Fourier space, for example, the weight function could be written as
#
# W^{s, XY} = \sum_{i=0}^{N} \int{d^2 z}
#                   (e^{+i*2\pi*s^{i,X}+i*(l_X.z)} W^{i,X}(l_X)) *
#                    (e^{+i*2\pi*s^{i,Y}+i*(l_Y.z)} W^{i,Y}(l_Y)) *
#                     (e^{-i*2\pi*s^{i,L}+i*( -L.z)} W^{i,L}( L ))
#
# where s^{i,X/Y/L} are integers representing a spin parameter for the
# components of the weight function. with such a weight function q(L)
# may be evaluated in O(N_i * lmax^2 * log(lmax)) using fast Fourier
# transforms (FFTs). a similar result holds for full-sky maps, using
# fast Spherical Harmonic Transforms (SHTs) to obtain an algorithm
# which is slightly slower, at O(N_i * lmax^3) algorithm. separable
# weight functions and spins are encapsulated by the 'qest' class.
#
# in addition to q^{XY}(L), we often want to evaluate
#
#   * the response of the estimator q^{XY}(L) to s(L). if the
#     filtering which relates \bar{X} to X is diagonal in
#     Fourier space with \bar{X}(L) = F(L)X(L) then this
#     can also be calculated quickly with FFTs on the flat-sky,
#     or Wigner D-matrices on the full-sky. these calculations
#     are performed by the 'qest.fill_resp' method.
#
#   * the ensemble-averaged cross-power <q1^{XY}(L) q2^{*ZA}(L)>,
#     given estimates of the spectral cross-powers
#     <\bar{X}(l)\bar{Z}^*(l)>, <\bar{X}(l),\bar{A}^*(l)>, etc.
#     again, this cross-spectrum can be calculated quickly using
#     FFTs or D-matrices. this calculation is performed by the
#     'qest.fill_clqq' method.
#

import numpy as np

from .. import shts
from .. import maps
from .. import math

def qe_cov_fill_helper( qeXY, qeZA, ret, fX, fY, *args, **kwargs ):
    if maps.is_cfft(ret):
        return qe_cov_fill_helper_flatsky( qeXY, qeZA, ret, fX, fY, *args, **kwargs )
    else:
        return qe_cov_fill_helper_fullsky( qeXY, qeZZ, ret, fX, fY, *args, **kwargs )

def qe_cov_fill_helper_flatsky( qeXY, qeZA, ret, fX, fY, switch_ZA=False, conj_ZA=False, npad=2):
    """ helper function to calculate various ensemble-average cross-products between two estimators qe1 and qe2. with all
    boolean options set to false this function calculates

        ret(L) = 0.25 * \int{d^2 l_X} \int{d^2 l_Y} fX(l_X) fY(l_Y) \sum_{ij} W_{XY}^{i}(l_X, l_Y, L) W_{ZA}^{j}(l_X, l_Y, L).

    where W_{XY} and W_{ZA} are weight function coming from the two estimators. options:
    
         * qeXY      = first estimator.
         * qeZA      = second estimator.
         * ret       = complex Fourier transform (maps.cfft) object in which to store the results.
         * fX, fY    = complex Fourier transform (maps.cfft) objects representing the filter functions for the X and Y fields.
                       these can also be 1D spectra, in which case they are broadcast to annuli in 2D.
         * switch_ZA = change W_{ZA}^{j}(l_X, l_Y, L) -> W_{ZA}^{j, ZA}(l_Y, l_X, L) .
         * conj_ZA   = take the complex conjugate of the W_{ZA} weight function. W_{ZA}^{j} -> W^{ZA}^{* j}.
    """
    assert( (npad == int(npad)) and (npad >= 1 ) )

    lx, ly = ret.get_lxly()
    l      = np.sqrt(lx**2 + ly**2)
    psi    = np.arctan2(lx, -ly)
    nx, ny = l.shape

    if fX.shape != l.shape:
        assert( len(fX.shape) == 1 )
        fX = np.interp( l.flatten(), np.arange(0, len(fX)), fX, right=0 ).reshape(l.shape)

    if fY.shape != l.shape:
        assert( len(fY.shape) == 1 )
        fY = np.interp( l.flatten(), np.arange(0, len(fY)), fY, right=0 ).reshape(l.shape)

    i1_ZA, i2_ZA = { False : (0,1), True : (1,0) }[switch_ZA]
    cfunc_ZA = { False : lambda v : v, True : lambda v : np.conj(v) }[conj_ZA]

    for i in xrange(0, qeXY.ntrm):
        for j in xrange(0, qeZA.ntrm):
            term1 = (qeXY.wl[i][0](l, lx, ly) * cfunc_ZA(qeZA.wl[j][i1_ZA](l, lx, ly)) * fX * 
                  np.exp(+1.j*(qeXY.sl[i][0]+(-1)**(conj_ZA)*qeZA.sl[j][i1_ZA])*psi))
            term2 = (qeXY.wl[i][1](l, lx, ly) * cfunc_ZA(qeZA.wl[j][i2_ZA](l, lx, ly)) * fY *
                  np.exp(+1.j*(qeXY.sl[i][1]+(-1)**(conj_ZA)*qeZA.sl[j][i2_ZA])*psi))

            ret.fft[:,:] += ( math.convolve_padded(term1, term2, npad=npad) * 0.25 / (ret.dx * ret.dy) *
                             ( qeXY.wl[i][2](l, lx, ly) * cfunc_ZA(qeZA.wl[j][2](l, lx, ly)) *
                              ( np.exp(-1.j*(qeXY.sl[i][2]+(-1)**(conj_ZA)*qeZA.sl[j][2])*psi) ) ) )

    return ret

def qe_cov_fill_helper_fullsky( qeXY, qeZA, ret, fX, fY, switch_ZA=False, conj_ZA=False):
    """ a full-sky version of qe_cov_fill_helper_flatsky.
    
         * qeXY      = first estimator.
         * qeZA      = second estimator.
         * ret       = complex array in which to store results (the length of this array, lmax+1, defines the maximum multipole).
         * fX, fY    = 1D real arrays representing the filter functions for the X and Y fields.
         * switch_ZA = change W_{ZA}^{j}(l_X, l_Y, L) -> W_{ZA}^{j, ZA}(l_Y, l_X, L) .
         * conj_ZA   = take the complex conjugate of the W_{ZA} weight function. W_{ZA}^{j} -> W^{ZA}^{* j}.
    """

    lmax = len(ret)-1
    
    i1_ZA, i2_ZA = { False : (0,1), True : (1,0) }[switch_ZA]
    cfunc_ZA = { False : lambda v : v, True : lambda v : np.conj(v) }[conj_ZA]

    lmax_fX = len(fX)-1
    lmax_fY = len(fY)-1

    for i in xrange(0, qeXY.ntrm):
        for j in xrange(0, qeZA.ntrm):
            # l1 part
            tl1min = max(abs(qeXY.sl[i][0]), abs(qeZA.sl[j][i1_ZA]))
            tl1max = min( [qeXY.lmax, qeZA.lmax, lmax_fX] )

            cl1 = np.zeros( tl1max+1, dtype=np.complex )
            for tl1 in xrange(tl1min, tl1max+1):
                cl1[tl1] = qeXY.wl[i][0](tl1) * cfunc_ZA( qeZA.wl[j][i1_ZA](tl1) ) * (2.*tl1+1.) * fX[tl1]

            # l2 part
            tl2min = max(abs(qeXY.sl[i][1]), abs(qeZA.sl[j][i2_ZA]))
            tl2max = min( [qeXY.lmax, qeZA.lmax, lmax_fY] )

            cl2 = np.zeros( tl2max+1, dtype=np.complex )
            for tl2 in xrange(tl2min, tl2max+1):
                cl2[tl2] = qeXY.wl[i][1](tl2) * cfunc_ZA( qeZA.wl[j][i2_ZA](tl2) ) * (2.*tl2+1.) * fY[tl2]

            # transform l1 and l2 parts to position space
            glq = math.wignerd.gauss_legendre_quadrature( (tl1max + tl2max + lmax)/2 + 1 )
            gp1 = glq.cf_from_cl( qeXY.sl[i][0], -(-1)**(conj_ZA)*qeZA.sl[j][i1_ZA], cl1 )
            gp2 = glq.cf_from_cl( qeXY.sl[i][1], -(-1)**(conj_ZA)*qeZA.sl[j][i2_ZA], cl2 )

            # multiply and return to cl space
            clL = glq.cl_from_cf( lmax, qeXY.sl[i][2], -(-1)**(conj_ZA)*qeZA.sl[j][2], gp1 * gp2 )

            for L in xrange(0, lmax+1):
                ret[L] += clL[L] * qeXY.wl[i][2](L) * cfunc_ZA( qeZA.wl[j][2](L) ) / (32.*np.pi)

    return ret

class qest(object):
    """ base class for a quadratic estiamtor q^{XY}(L),
    which can be run on fields \bar{X} and \bar{Y} as

    q^{XY}(L) = 1/2 \int{d^2 l_X} \int{d^2 l_Y}
                    W^{XY}(l_X, l_Y, L) \bar{X}(l_X) \bar{Y}(l_Y)

    with l_X + l_Y = L.

    the weight function W^{s, XY} must be separable. for
    flat-sky calculation it is encoded as

    W^{XY} = \sum_{i=0}^{N} \int{d^2 z}
                    (e^{+i*2\pi*s^{i,X}+i*(l_X.z)} W^{i,X}(l_X)) *
                     (e^{+i*2\pi*s^{i,Y}+i*(l_Y.z)} W^{i,Y}(l_Y)) *
                      (e^{-i*2\pi*s^{i,L}+i*( -L.z)} W^{i,L}( L ))

    for full-sky calculations it is encoded as

    W^{XY} = \sum_{i=0}^{N_i} \int{d^2 n}
                    {}_s^{i,X}Y_{l_X m_X}(n) W^{i,X}(l_X) *
                     {}_s^{i,Y}Y_{l_Y m_Y}(n) W^{i,Y}(l_Y) *
                      {}_s^{i,L}Y_{  L M  }(n) W_^{i,L}( L ).

    the spins s^{i,n} are stored in an array self.s[i][n] and the
    weights w^{i,n}(l) are encapsulated as functions w[i][n](l). some
    flat-sky-specific weight functions may also take lx, ly as arguments.
    """
    def __init__(self):
        pass

    def eval( self, barX, barY=None, **kwargs ):
        if barY is None:
            barY = barX

        if False:
            pass
        elif maps.is_cfft(barX):
            assert( maps.is_cfft(barY) )
            return self.eval_flatsky( barX, barY, **kwargs )
        elif maps.is_rfft(barX):
            assert( maps.is_rfft(barX) )
            return self.eval_flatsky( barX, barY, **kwargs )
        else:
            return self.eval_fullsky( barX, barY, **kwargs )

    def eval_fullsky( self, barX, barY ):
        """ evaluate this quadratic estimator on the full-sky, returning

        q^{XY}(L) = 1/2 \sum_{l_X} \sum_{l_Y}
                     W^{XY}(l_X, l_Y, L) \bar{X}(l_X) \bar{Y}(l_Y)

        where L, l_X and l_Y represent (l,m) spherical harmonic modes.

        inputs:
             * barX            = input field \bar{X}. should be 2D complex array
                                 containing the harmonic modes for the X field
                                 (in the 'vlm' indexing scheme, see shts.util).
             * barY            = input field \bar{Y}. should be 2D complex array
                                 containing the harmonic modes for the X field
                                 (in the 'vlm' indexing scheme, see shts.util).
        """
        lmax   = self.lmax
        lmax_X = shts.util.nlm2lmax( len(barX) )
        lmax_Y = shts.util.nlm2lmax( len(barY) )

        nphi   = lmax_X+lmax_Y+lmax+1
        glq    = math.wignerd.gauss_legendre_quadrature( (lmax_X + lmax_Y + lmax)/2 + 1 )
        tht    = np.arccos(glq.zvec)
        phi    = np.linspace(0., 2.*np.pi, nphi, endpoint=False)

        ret = np.zeros( (lmax+1)**2, dtype=np.complex )
        for i in xrange(0, self.ntrm):
            # l_X term
            vlx = shts.util.alm2vlm( barX )
            for l in xrange(0, lmax_X+1):
                vlx[l**2:(l+1)**2] *= self.get_wlX(i,l)
            vmx = shts.vlm2map( self.get_slX(i), tht, phi, vlx )
            del vlx

            # l_Y term
            vly = shts.util.alm2vlm( barY )
            for l in xrange(0, lmax_X+1):
                vly[l**2:(l+1)**2] *= self.get_wlY(i,l)
            vmy = shts.vlm2map( self.get_slY(i), tht, phi, vly )
            del vly

            # multiply in position space
            vmm  = vmx * vmy
            del vmx, vmy

            # apply weights for harmonic integration
            for j, w in enumerate(glq.wvec):
                vmm[j,:] *= w * (2.*np.pi / nphi)

            # perform integration
            vlm = shts.map2vlm(lmax, self.get_slL(i), tht, phi, vmm)
            del vmm

            for l in xrange(0, lmax+1):
                vlm[l**2:(l+1)**2] *= 0.5*self.get_wlL(i,l)

            ret += vlm

        return ret
    
    def eval_flatsky( self, barX, barY, npad=2 ):
        """ evaluate this quadratic estimator on the flat-sky, returning

        q^{XY}(L) = 1/2 \int{d^2 l_X} \int{d^2 l_Y}
                        W^{XY}(l_X, l_Y, L) \bar{X}(l_X) \bar{Y}(l_Y)

        where L, l_X and l_Y represent modes in 2D Fourier space.

        inputs:
             * barX            = input field \bar{X}. should be 2D complex
                                 Fourier transform (maps.cfft) object, or have
                                 a get_cfft() method so that it can be converted.
             * barY            = input field \bar{X}. should be 2D complex
                                 Fourier transform (maps.cfft) object, or have
                                 a get_cfft() method so that it can be converted.
             * (optional) npad = padding factor to avoid aliasing in the convolution.
        """
        if hasattr(barX, 'get_cfft'):
            barX = barX.get_cfft()
        if hasattr(barY, 'get_cfft'):
            barY = barY.get_cfft()
        assert( barX.compatible(barY) )

        cfft   = maps.cfft( barX.nx, barX.dx, ny=barX.ny, dy=barX.dy )

        lx, ly = cfft.get_lxly()
        l      = np.sqrt(lx**2 + ly**2)
        psi    = np.arctan2(lx, -ly)

        fft = cfft.fft

        for i in xrange(0, self.ntrm):
            term1 = self.wl[i][0](l, lx, ly) * barX.fft * np.exp(+1.j*self.sl[i][0]*psi) 
            term2 = self.wl[i][1](l, lx, ly) * barY.fft * np.exp(+1.j*self.sl[i][1]*psi) 

            fft[:,:] += ( math.convolve_padded(term1, term2, npad=npad ) *
                         ( self.wl[i][2](l, lx, ly) * np.exp(-1.j*self.sl[i][2]*psi) ) *
                           ( 0.5 / np.sqrt(cfft.dx * cfft.dy) * np.sqrt(cfft.nx * cfft.ny) ) )

        return cfft

    def fill_resp( self, qeZA, ret, fX, fY, **kwargs ):
        if maps.is_cfft(ret):
            return self.fill_resp_flatsky( qeZA, ret, fX, fY, **kwargs )
        else:
            return self.fill_resp_fullsky( qeZA, ret, fX, fY, **kwargs )

    def fill_resp_flatsky( self, qeZA, ret, fX, fY, npad=2 ):
        """ compute the response of this estimator to the statistical
        anisotropy encapsulated by a second estimator qeZA,
        
            R(L) = 1/2 \int{d^2 l_X} \int{d_2 l_Y}
                         W^{XY} W^{ZA} fX(l_X) fY(l_Y).

        with l_X+l_Y=L and fX(l_X) fY(l_y) represent filters which are
        diagonal in Fourier space applied to the X and Y fields.

        dividing the output of self.eval() by this response gives a properly
        normalized estimator for the statistical anisotropy defined by qeZA.
        """
        ret.fft[:,:] = 0.0
        qe_cov_fill_helper_flatsky( self, qeZA, ret, fX, fY, switch_ZA=False, conj_ZA=False, npad=npad)
        ret.fft[:,:] *= 2.0 # multiply by 2 because qe_cov_fill_helper returns 1/2 the response.
        return ret

    def fill_resp_fullsky( self, qeZA, ret, fX, fY, npad=2 ):
        """ compute the response of this estimator to the statistical
        anisotropy encapsulated by a second estimator qeZA,
        
            R(L) = 1/2 \int{d^2 l_X} \int{d_2 l_Y}
                         W^{XY} W^{ZA} fX(l_X) fY(l_Y).

        with l_X+l_Y=L and fX(l_X) fY(l_y) represent filters which are
        diagonal in Fourier space applied to the X and Y fields.

        dividing the output of self.eval() by this response gives a properly
        normalized estimator for the statistical anisotropy defined by qeZA.
        """
        ret[:] = 0.0
        qe_cov_fill_helper_fullsky( self, qeZA, ret, fX, fY, switch_ZA=False, conj_ZA=False)
        ret[:] *= 2.0 # multiply by 2 because qe_cov_fill_helper returns 1/2 the response.
        return ret

    def fill_clqq( self, ret, fXX, fXY, fYY, **kwargs ):
        if maps.is_cfft(ret):
            return self.fill_clqq_flatsky( ret, fXX, fXY, fYY, **kwargs )
        else:
            return self.fill_clqq_fullsky( ret, fXX, fXY, fYY, **kwargs )
    
    def fill_clqq_flatsky( self, ret, fXX, fXY, fYY, npad=2):
        """ compute the ensemble-averaged auto-power < |q^{XY}(L)[\bar{X}, \bar{Y}]|^2 >,
        given estimates of the auto- and cross-spectra of \bar{X} and \bar{Y}.

             * ret             = complex Fourier transform (maps.cfft) object defining the
                                 pixelization on which to evaluate the auto-power.
             * fXX             = estimate of <\bar{X} \bar{X}^*>
             * fXY             = estimate of <\bar{X} \bar{Y}^*>
             * fYY             = estimate of <\bar{Y} \bar{Y}^*>
             * (optional) npad = padding factor to avoid aliasing in the convolution.
        """
        ret.fft[:,:] = 0.0
        qe_cov_fill_helper_flatsky( self, self, ret, fXX, fYY, switch_ZA=False, conj_ZA=True, npad=npad )
        qe_cov_fill_helper_flatsky( self, self, ret, fXY, fXY, switch_ZA=True,  conj_ZA=True, npad=npad )
        return ret

    def fill_clqq_fullsky( self, ret, fXX, fXY, fYY):
        """ compute the ensemble-averaged auto-power < |q^{XY}(L)[\bar{X}, \bar{Y}]|^2 >,
        given estimates of the auto- and cross-spectra of \bar{X} and \bar{Y}.

             * ret             = complex numpy array whose length (lmax+1) defines the maximum multipole
                                 at which to evaluate the auto-power.
             * fXX             = estimate of <\bar{X} \bar{X}^*>
             * fXY             = estimate of <\bar{X} \bar{Y}^*>
             * fYY             = estimate of <\bar{Y} \bar{Y}^*>
        """
        ret[:] = 0.0
        qe_cov_fill_helper_fullsky( self, self, ret, fXX, fYY, switch_ZA=False, conj_ZA=True )
        qe_cov_fill_helper_fullsky( self, self, ret, fXY, fXY, switch_ZA=True,  conj_ZA=True )
        return ret

    def get_slX(self, i):
        return self.sl[i][0]

    def get_slY(self, i):
        return self.sl[i][1]

    def get_slL(self, i):
        return self.sl[i][2]

    def get_wlX(self, i, l, **kwargs):
        return self.wl[i][0](l, **kwargs)

    def get_wlY(self, i, l, **kwargs):
        return self.wl[i][1](l, **kwargs)

    def get_wlL(self, i, l, **kwargs):
        return self.wl[i][2](l, **kwargs)
