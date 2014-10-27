# quicklens/sims/sims.py
# --
# this module contains code for simulating lensed flat-sky and curved-sky cmb maps, as well as
# semi-realistic instrumental noise maps. it is built around 'library' classes which
# provide an interface to data and simulations, either of the maps themselves or
# after the result of some set of filtering operations. they usually have methods such as
#
#  * get_[dat or sim]_tqu (for simulated maps)
#  * get_[dat or sim]_teb (for simulated Fourier modes, or filtered sky maps).
#
# library objects which perform expensive, time-consuming operations on their inputs often
# cache results to a directory. to sanity-check that their parameters or inputs have not changed
# since the caching was performed, they have a 'hashdict()' method which returns a dictionary
# intended to uniquely characterize each library object.
#

import os, copy
import numpy as np

from .. import maps
from .. import spec

class library_tqu_sum():
    """ helper library for combining a list of [tqumap] libraries into a sum. if data is requested, it is pulled from the first library. """
    def __init__(self, cpts):
        """ initialize the library_tqu_sum. cpts is a list of tqumap libraries. """
        assert( len(cpts) > 0 )
        self.cpts = cpts
    def hashdict(self):
        return { i : cpt.hashdict() for i, cpt in enumerate(self.cpts) }
    def get_dat_tqu(self):
        return self.cpts[0].get_dat_tqu()
    def get_sim_tqu(self,idx):
        ret = self.cpts[0].get_sim_tqu(idx)
        for cpt in self.cpts[1:]:
            ret += cpt.get_sim_tqu(idx)
        return ret

def tqumap_homog_noise( pix, nlev_t, nlev_p):
    """ generate a maps.tqumap object containing homogeneous white noise which is uncorrelated between pixels.
         * pix    = pixelization object defining the map size and scale (of type maps.pix).
         * nlev_t = noise level of the temperature map (in uK.arcmin).
         * nlev_p = noise level of the polarization map (in uK.arcmin).
    """
    nx, dx, ny, dy = pix.nx, pix.dx, pix.ny, pix.dy
    
    ret = maps.tqumap( nx, dx, ny=ny, dy=dy )
    ret.tmap += np.random.standard_normal(ret.tmap.shape) * nlev_t / (180.*60./np.pi*np.sqrt(ret.dx * ret.dy))
    ret.qmap += np.random.standard_normal(ret.qmap.shape) * nlev_p / (180.*60./np.pi*np.sqrt(ret.dx * ret.dy))
    ret.umap += np.random.standard_normal(ret.umap.shape) * nlev_p / (180.*60./np.pi*np.sqrt(ret.dx * ret.dy))
    return ret

def tebfft(pix, tcl):
    """ generate a maps.tebfft object containing a simulated sky map.
         * pix    = pixelization object defining the map size and scale (of type maps.pix).
         * tcl    = an object with 'lmax' attribute as well as 'cltt', 'clee', 'clte', etc. attributes for all of the non-zero ensemble-averaged auto- and cross-spectra in the simulation.
    """
    nx, dx, ny, dy = pix.nx, pix.dx, pix.ny, pix.dy
        
    tfft      = (np.random.standard_normal((ny,nx/2+1)) + 1.j * np.random.standard_normal((ny,nx/2+1))) / np.sqrt(2.)
    efft      = (np.random.standard_normal((ny,nx/2+1)) + 1.j * np.random.standard_normal((ny,nx/2+1))) / np.sqrt(2.)
    bfft      = (np.random.standard_normal((ny,nx/2+1)) + 1.j * np.random.standard_normal((ny,nx/2+1))) / np.sqrt(2.)

    tfft[0,0] = np.sqrt(2.) * np.real(tfft[0,0])
    efft[0,0] = np.sqrt(2.) * np.real(efft[0,0])
    bfft[0,0] = np.sqrt(2.) * np.real(bfft[0,0])

    teb       = maps.tebfft( nx, dx, [tfft, efft, bfft], ny=ny, dy=dy )
    return spec.clmat_teb(tcl).cholesky() * teb

def rfft(pix, tcl):
    """ generate a maps.rfft object containing a simulated sky map.
         * pix    = pixelization object defining the map size and scale (of type maps.pix).
         * tcl    = a vector containing the ensemble-average power spectrum for the simulated map.
    """
    nx, dx, ny, dy = pix.nx, pix.dx, pix.ny, pix.dy
    
    ret = maps.rfft(nx, dx, ny=ny, dy=dy)
    ret.fft[:,:] = (np.random.standard_normal((ny,nx/2+1)) + 1.j * np.random.standard_normal((ny,nx/2+1))) / np.sqrt(2.)
    ret.fft[0,0] = np.sqrt(2.) * np.real(ret.fft[0,0])
    ret.fft[ny/2+1:, 0] = np.conj( ret.fft[1:ny/2,0][::-1] )

    ret.fft[:,:] *= np.interp( ret.get_ell().flatten(), np.arange(0,len(tcl)), np.sqrt(tcl), right=0 ).reshape( ret.fft.shape )
    return ret

