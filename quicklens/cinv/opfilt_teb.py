# quicklens/cinv/opfilt_teb.py
# --
# operations and filters for applying the flat-sky temperature+polarization
# C-inverse operation. we model a map 'd' as a vector of pixel values for T, Q, U
# given by
#
# d = Y * s + n
#
# where
#
#  * 's' is a vector containing the Fourier modes of the sky in the map region.
#  * 'Y' is the 'pointing matrix', which applies an instrumental transfer function
#        to 's' and then transforms it into pixel space.
#  * 'n' is a map containing the instrumental noise realization in each pixel.
#
# in this model, the 'C-inverse' or filter is used to remove the transfer function
# and downweight noisy modes (see  http://arxiv.org/pdf/astro-ph/9805339 for a discussion)
#
# C^{-1} * d = [ (Y * N * Y'^{-1}) + S ]^{-1} * Y^{-1} * d
#
# where
#  * 'N' is the pixel-pixel covariance matrix <n n'>, where <> represents ensemble-averaging.
#  * 'S' is the mode-mode signal covariance matrix <s s'> where <> represents ensemble averaging.
#
# C^{-1} * d corresponds to Fourier-transforming the data map, deconvolving the transfer function
# and then multiplying by the inverse of the signal+noise covariance matrix.
#
# in this module, the C^{-1} filter is implemented in the form
#
# C^{-1} * d = S^{-1} [S^{-1} + (Y' * N^{-1} * Y)]^{-1} * Y' * N^{-1} * d
#
# this allows pixels to be masked by taking the noise level to infinity
# (so that N^{-1} -> 0 for masked pixels).
# 
# this module contains classes representing S^{-1}, (Y' * N^{-1} * Y), and (Y' * N^{-1}),
# and for multiplication by the []^{-1} term using conjugate descent.

import os
import numpy  as np

from .. import util
from .. import spec
from .. import maps

class sinv_filt(spec.clmat_teb):
    """ class representing the S^{-1} filter in the C-inverse operation. """
    def __init__(self, cl):
        """ initialize the S^{-1} filtering.
             * cl = object with cltt, clte, etc. attributes for the non-zero signal spectra.
        """
        super(sinv_filt, self).__init__(cl)
        self.clmat = self.inverse().clmat

    def hashdict(self):
        return { 'lmax'  : self.lmax,
                 'clmat' : self.clmat }

class ninv_filt():
    """ class representing the (Y' * N^{-1} * Y) and (Y' * N^{-1}) filters in the C-inverse operation. """
    def __init__(self, transf, ninv):
        """ initialize the (Y' * N^{-1} * Y) and (Y' * N^{-1}) filters.
             transf = object representing the beam and filter transfer function in Fourier space (can be any object with a multiplication operation for maps.tebfft).
             ninv   = object representing the pixel-uncorrelated noise covariance matrix in each pixel of the map (typically a maps.tqumap or maps.tqumap_wt object).
        """
        self.transf   = transf
        self.ninv     = ninv

    def hashdict(self):
        ret = {}

        if type(self.transf) == np.ndarray:
            ret['transf'] = self.transf
        else:
            ret['transf'] = self.transf.hashdict()

        ret['ninv'] = self.ninv.hashdict()

        return ret

    def degrade(self, tfac):
        """ returns a copy of this filter object appropriate for a map with half the resolution / number of this pixels. """
        return ninv_filt( self.transf, self.ninv.degrade(tfac, intensive=False) )

    def mult_tqu(self, tqu):
        """ returns Y' * N^{-1} * tqu, where tqu is a maps.tqumap object. """
        assert(self.ninv.compatible(tqu))

        ret  = (tqu * self.ninv).get_teb() * self.transf

        ret *= (1. / (ret.dx * ret.dy))
        return ret

    def mult_teb(self, teb):
        """ returns (Y' * N^{-1} * Y) * teb, where teb is a maps.tebfft object. """
        assert( maps.pix.compatible( self.ninv, teb ) )
        
        ret = ( (teb * self.transf).get_tqu() * self.ninv ).get_teb() * self.transf
        ret *= (1. / (ret.dx * ret.dy))
        return ret

    def get_lmax(self):
        """ return the maximum multipole of the pointing matrix Y. """
        if (False):
            pass
        elif hasattr(self.transf, 'get_ell'):
            return np.ceil( np.max( self.transf.get_ell().flatten() ) )
        elif hasattr(self.transf, 'lmax'):
            return self.transf.lmax
        elif ( ( getattr(self.transf, 'size', 0) > 1 ) and ( len( getattr(self.transf, 'shape', ()) ) == 1 ) ):
            return len(self.transf)-1
        else:
            assert(0)

    def get_fmask(self):
        """ return a 2D matrix which is set to zero for all pixels where N^{-1} == 0 (and 1 elsewhere). """
        if (False):
            pass
        elif maps.is_tqumap_wt(self.ninv):
            mskt = np.flatnonzero( self.ninv.weight[:,:,0,0] )
            mskq = np.flatnonzero( self.ninv.weight[:,:,1,1] )
            msku = np.flatnonzero( self.ninv.weight[:,:,2,2] )

            assert( np.all(mskt == mskq) )
            assert( np.all(mskt == msku) )
            
            ret = np.zeros( self.ninv.nx * self.ninv.ny ); ret[mskt] = 1.
            return ret.reshape( (self.ninv.nx, self.ninv.ny) )
        elif maps.is_tqumap(self.ninv):
            mskt = np.flatnonzero( self.ninv.tmap )
            mskq = np.flatnonzero( self.ninv.qmap )
            msku = np.flatnonzero( self.ninv.umap )

            assert( np.all(mskt == mskq) )
            assert( np.all(mskt == msku) )

            ret = np.zeros( self.ninv.nx * self.ninv.ny ); ret[mskt] = 1.
            return ret.reshape( (self.ninv.nx, self.ninv.ny) )
        else:
            assert(0)

    def get_fcut(self):
        """ return the fraction of the map which is not completely masked by N^{-1}. """
        fmask = self.get_fmask()
        return np.sum(fmask.flatten())/fmask.size

    def get_fl(self):
        """ return an approximation of [Y' N^{-1} Y] which is diagonal in Fourier space. """
        if (False):
            pass
        elif maps.is_tqumap_wt(self.ninv):
            lmax = self.get_lmax()
            fcut = self.get_fcut()

            ntt = np.average(self.ninv.weight[:,:,0,0]) / fcut
            nee = 0.5*(np.average(self.ninv.weight[:,:,1,1]) + np.average(self.ninv.weight[:,:,2,2])) / fcut
            
            ninv = spec.clmat_teb( util.dictobj( { 'lmax' : lmax,
                                                   'cltt' : ntt * np.ones(lmax+1),
                                                   'clee' : nee * np.ones(lmax+1),
                                                   'clbb' : nee * np.ones(lmax+1) } ) )

            return ninv * self.transf * self.transf * (1. / (self.ninv.dx * self.ninv.dy))
        elif maps.is_tqumap(self.ninv):
            lmax = self.get_lmax()
            fcut = self.get_fcut()

            ntt = np.average(self.ninv.tmap)/fcut
            nee = 0.5*(np.average(self.ninv.qmap) + np.average(self.ninv.umap))/fcut
            
            ninv = spec.clmat_teb( util.dictobj( { 'lmax' : lmax,
                                                   'cltt' : ntt * np.ones(lmax+1),
                                                   'clee' : nee * np.ones(lmax+1),
                                                   'clbb' : nee * np.ones(lmax+1) } ) )

            return ninv * self.transf * self.transf * (1. / (self.ninv.dx * self.ninv.dy))
        else:
            assert(0)

# ===

def calc_prep(tqu, sinv_filt, ninv_filt):
    """ preparation operation: returns the result of multipling tmap by Y^{t} N^{-1}. """
    return ninv_filt.mult_tqu(tqu)

def calc_fini(teb, sinv_filt, ninv_filt):
    """ finalization operation: returns the resuult of multiplying tfft by S^{-1}. """
    return sinv_filt * teb

# ===

class dot_op():
    """ defines a dot product for two maps.tebfft objects using their cross-spectra up to a specified lmax. """
    def __init__(self, lmax=None):
        self.lmax = lmax

    def __call__(self, teb1, teb2):
        assert( teb1.compatible(teb2) )

        if self.lmax != None:
            lmax = self.lmax

            return np.sum( ( teb1.tfft * np.conj(teb2.tfft) +
                             teb1.efft * np.conj(teb2.efft) +
                             teb1.bfft * np.conj(teb2.bfft) ).flatten()[np.where(teb1.get_ell().flatten() <= lmax)].real )
        
        else:
            return np.sum( ( teb1.tfft * np.conj(teb2.tfft) +
                             teb1.efft * np.conj(teb2.efft) +
                             teb1.bfft * np.conj(teb2.bfft) ).flatten().real )

class fwd_op():
    """ returns [S^{-1} + (Y' * N^{-1} * Y)] * teb. """
    def __init__(self, sinv_filt, ninv_filt):
        self.sinv_filt = sinv_filt
        self.ninv_filt = ninv_filt

    def __call__(self, teb):
        return self.calc(teb)

    def calc(self, teb):
        return (self.sinv_filt * teb + self.ninv_filt.mult_teb(teb))

# ===

class pre_op_diag():
    """ returns an approximation of the operation [Y^{t} N^{-1} Y + S^{-1}]^{-1} which is diagonal in Fourier space, represented by a maps.tebfft object. """
    def __init__(self, sinv_filt, ninv_filt):
        self.filt = (sinv_filt + ninv_filt.get_fl()).inverse()

    def __call__(self, talm):
        return self.calc(talm)
        
    def calc(self, teb):
        return self.filt * teb
