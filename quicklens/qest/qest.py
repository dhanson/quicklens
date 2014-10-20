# quicklens/qest/qest.py
# --
# this module contains classes and routines for applying quadratic
# anisotropy estimators to flat-sky maps.
#
# these estimators are motivated as follows. first, we start with
# a source of statistical anisotropy 's', which is a 2D field having
# a Fourier transform s(L). 's' induces couplings between modes of the
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
# where \bar{X} and \bar{Y} are filtered observations of the X and Y fields,
# and l_X+l_Y = L. Nominally, the cost of evaluating q(L) is O(lmax^{6})
# (because l_X, l_Y, and L are each 2D fields with lmax^2 elements), however
# if the weight function as a sum of terms which have separable dependence
# on l_X, l_Y, and L, with the form
#
# W^{s, XY} = \sum_{i=0}^{N_i} (e^{i*2\pi*s^{i,X}} * W^{i,X}(l_X)) *
#                               (e^{i*2\pi*s^{i,Y}} * W^{i,Y}(l_Y)) *
#                                (e^{-i*2\pi*s^{i,L}} * W^{i,L}(L))
#
# where s^{i,X/Y/L} are integers representing a spin parameter for each
# component of the weight function. with such a weight function q(L)
# may be evaluated in O(N_i * lmax^2 * log(lmax)) using fast Fourier
# transforms (FFTs). in this module such weight functions and spins
# are encapsulated by the 'qest' class.
#
# in addition to q^{XY}(L), one often wants to evaluate
#
#   * the response of the estimator q^{XY}(L) to s(L). if the
#     filtering which relates \bar{X} to X is diagonal in
#     Fourier space with \bar{X}(L) = F(L)X(L) then this
#     can also be calculated quickly with FFTs, and is
#     encapsulated in 'qest.fill_resp' method.
#
#   * the ensemble-averaged cross-power <q1^{XY}(L) q2^{*ZA}(L)>,
#     given estimates of the spectral cross-powers
#     <\bar{X}(l)\bar{Z}^*(l)>, <\bar{X}(l),\bar{A}^*(l)>, etc.
#     again, this cross-spectrum can be calculated quickly using FFTs.
#     it is encapsulated in the 'qest.fill_clqq' method.
#

import numpy as np

from .. import maps
from .. import math

def qe_cov_fill_helper( qeXY, qeZA, ret, fX, fY, switch_ZA=False, conj_ZA=False, npad=2):
    """ helper function to calculate various ensemble-average cross-products between two estimators qe1 and qe2. with all
    boolean options set to false this function calculates

        ret(L) = 0.25 * \int{d^2 l_X} \int{d^2 l_Y} fX(l_X) fY(l_Y) \sum_{ij} W_{XY}^{i}(l_X, l_Y, L) W_{ZA}^{j}(l_X, l_Y, L) .

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


class qest():
    """ base class for a quadratic estiamtor q^{XY}(L),
    which can be run on fields \bar{X} and \bar{Y} as

    q^{XY}(L) = 1/2 \int{d^2 l_X} \int{d^2 l_Y}
                    W^{s, XY}(l_1, l_2, L) \bar{X}(l_X) \bar{Y}(l_Y)

    with l_X + l_Y = L.

    the weight function W^{s, XY} must be separable, and is
    encoded as

    W^{s,XY} = \sum_{i=0}^{N_i} (e^{i*2\pi*s^{i,X}} * W^{i,X}(l_X)) *
                                 (e^{i*2\pi*s^{i,Y}} * W^{i,Y}(l_Y)) *
                                  (e^{-i*2\pi*s^{i,L}} * W^{i,L}(L)).

    the spins s^{i,n} are stored in an array self.s[i][n] and the
    weights w^{i,n}(l) are encapsulated as functions w[i][n](l,l_x,l_y). 
    """
    def __init__(self):
        pass

    def eval( self, barX, barY, npad=2 ):
        """ evaluate this quadratic estimator, returning

        q^{XY}(L) = 1/2 \int{d^2 l_X} \int{d^2 l_Y}
                        W^{XY}(l_1, l_2, L) \bar{X}(l_X) \bar{Y}(l_Y)

        inputs:
             * barX            = input field \bar{X}. should be 2D complex Fourier transform (maps.cfft) object, or have a get_cfft() method to convert.
             * barY            = input field \bar{X}. should be 2D complex Fourier transform (maps.cfft) object, or have a get_cfft() method to convert.
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

    def fill_resp( self, qeZA, ret, fX, fY, npad=2 ):
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
        qe_cov_fill_helper( self, qeZA, ret, fX, fY, npad=npad)
        ret.fft[:,:] *= 2.0 # multiply by 2 because qe_cov_fill_helper returns 1/2 the response.
        return ret

    def fill_clqq( self, ret, fXX, fXY, fYY, npad=2):
        """ compute the ensemble-averaged auto-power < |q^{XY}(L)[\bar{X}, \bar{Y}]|^2 >,
        given estimates of the auto- and cross-spectra of \bar{X} and \bar{Y}.

             * ret             = complex Fourier transform (maps.cfft) object defining the pixelization on which to evaluate the auto-power.
             * fXX             = estimate of <\bar{X} \bar{X}^*>
             * fXY             = estimate of <\bar{X} \bar{Y}^*>
             * fYY             = estimate of <\bar{Y} \bar{Y}^*>
             * (optional) npad = padding factor to avoid aliasing in the convolution.
        """
        ret.fft[:,:] = 0.0
        qe_cov_fill_helper( self, self, ret, fXX, fYY, switch_ZA=False, conj_ZA=True, npad=npad )
        qe_cov_fill_helper( self, self, ret, fXY, fXY, switch_ZA=True,  conj_ZA=True, npad=npad )
        return ret

    def get_slX(self, i):
        return self.sl[i][0]

    def get_slY(self, i):
        return self.sl[i][1]

    def get_slL(self, i):
        return self.sl[i][2]

    def get_wlX(self, i, l, lx, ly):
        return self.wl[i][0](l, lx, ly)

    def get_wlY(self, i, l, lx, ly):
        return self.wl[i][1](l, lx, ly)

    def get_wlL(self, i, l, lx, ly):
        return self.wl[i][2](l, lx, ly)
