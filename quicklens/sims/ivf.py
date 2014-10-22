# quicklens/sims/ivf.py
# --
# this module contains libraries for inverse-variance filtering sky maps.
# each library implements several methods:
#    get_sim_teb(i)  -- returns a maps.tebfft object containing the inverse-variance
#                       filtered Fourier modes for simulation i.
#    get_fmask()     -- returns a 2d matrix or maps.rmap object containing zeros
#                       for pixels which are masked by the filtering.
#    get_fl()        -- returns a maps.tebfft object 'fl' representing an approximation
#                       to the filter library which is diagonal in Fourier space.
#                       this can be used for analytical calculations involving filtered maps,
#                       the inverse-variance Fourier modes \bar{s} returned by get_sim_teb(i) can
#                       be modelled as \bar{s} = fl * s + n, where 's' is the sky signal and
#                       n is the filtered noise realization.

import os, sys, hashlib
import numpy  as np
import pickle as pk

import quicklens as ql
import util

class library(object):
    """ base class for inverse-variance filtered objects. """
    def __init__(self, obs_lib, lib_dir=None):
        """ initialize the inverse-variance filter.
                obs_lib            = library object (likely from sims.obs) which a get_sim_tqu(i) method for returning the map for simulation i.
                (optional) lib_dir = directory to store the hash of this object and likely cache files as well.
        """
        self.obs_lib = obs_lib
        
        if lib_dir != None:
            if ql.mpi.rank == 0:
                if not os.path.exists(lib_dir):
                    os.makedirs(lib_dir)

                if not os.path.exists(lib_dir + "/sim_hash.pk"):
                    pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
            ql.mpi.barrier()
            util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def hashdict(self):
        """ return a dictionary uniquely associated with the contents of this library. """
        return { 'obs_lib' : self.obs_lib.hashdict() }

    def get_sim_teb(self, i):
        """ return a maps.tebfft object containing the inverse-variance filtered Fourier modes for simulation i. """
        assert(0) # implement in subclass
        
    def get_sim_t(self, i):
        """ return a maps.cfft object containing the temperature component of get_sim_teb(i). """
        return self.get_sim_teb(i).get_cffts()[0]
    def get_sim_e(self, idx):
        """ return a maps.cfft object containing the e-mode component of get_sim_teb(i). """
        return self.get_sim_teb(i).get_cffts()[1]
    def get_sim_b(self, idx):
        """ return a maps.cfft object containing the b-mode component of get_sim_teb(i). """
        return self.get_sim_teb(i).get_cffts()[2]

    def get_fmask(self):
        """ return the mask map associated with this inverse-variance filter. """
        assert(0) # implement in subclass.

    def get_fl(self):
        """ return a maps.tebfft object representing an approximation
        to the filter library which is diagonal in Fourier space.
        this can be used for analytical calculations involving filtered maps. """
        assert(0) # implement in subclass.

    def get_flt(self):
        """ return the temperature component of get_fl(). """
        return self.get_fl().get_cffts()[0]

    def get_fle(self):
        """ return the e-mode component of get_fl(). """
        return self.get_fl().get_cffts()[1]

    def get_flb(self):
        """ return the b-mode component of get_fl(). """
        return self.get_fl().get_cffts()[2]

class library_diag(library):
    """ a simple inverse-variance filter which is (nearly) diagonal in Fourier space. the steps are:
            1) apply a mask to the observed map.
            2) take the Fourier transform.
            3) deconvolve the beam+filtering+pixelization transfer function.
            4) divide by the sky+noise power spectrum in T, E, B for white noise with a given level.
    """
    def __init__(self, obs_lib, cl, transf, nlev_t=0., nlev_p=0., mask=None):
        """ initialize the diagonal filter library.
                obs_lib = library object (likely from sims.obs) which a get_sim_tqu(i) method for returning the map for simulation i.
                cl      = spec.clmat_teb object containing the theory sky power spectra.
                transf  = 1d array or maps.tebfft object containing the beam+filtering+pixelization transfer function.
                (optional) nlev_t = temperature map white noise level to use in filtering (in uK.arcmin).
                (optional) nlev_p = polarization (q,u) map white noise level to use in filtering (in uK.arcmin).
                (optional) mask   = 2d matrix or maps.rmap object to use as a mask. defaults to no masking.
        """
        self.cl     = cl
        self.transf = transf
        self.nlev_t = nlev_t
        self.nlev_p = nlev_p
        
        self.mask   = mask
        if self.mask == None:
            self.mask = np.ones( (obs_lib.pix.ny, obs_lib.pix.nx) )

        self.nl = ql.spec.cl2tebfft(ql.util.dictobj( {'lmax' : cl.lmax,
                                                      'cltt' : np.ones(cl.lmax+1) * (nlev_t * np.pi/180./60.)**2,
                                                      'clee' : np.ones(cl.lmax+1) * (nlev_p * np.pi/180./60.)**2,
                                                      'clbb' : np.ones(cl.lmax+1) * (nlev_p * np.pi/180./60.)**2} ), obs_lib.pix )

        self.tl = self.transf.inverse()
        self.fl = (self.cl + (self.tl * self.tl * self.nl)).inverse()
        
        super(library_diag, self).__init__(obs_lib)

    def hashdict(self):
        ret = { 'cl'     : self.cl.hashdict(),
                'transf' : self.transf.hashdict(),
                'nlev_t' : self.nlev_t,
                'nlev_p' : self.nlev_p,
                'super'  : super(library_diag, self).hashdict() }
        if type(self.mask) == np.ndarray:
            ret['mask'] = hashlib.sha1(self.mask.view(np.uint8)).hexdigest()
        else:
            ret['mask'] = self.mask.hashdict()

    def get_fmask(self):
        return self.mask
            
    def get_fl(self):
        return self.fl

    def get_sim_teb(self, idx):
        ret = (self.obs_lib.get_sim_tqu(idx) * self.mask).get_teb() * self.tl
        return ret * self.get_fl()

class library_l_mask(library):
    """ a simple wrapper around another inverse-variance filter library which applies a multipole mask in Fourier space. """
    def __init__(self, ivf_lib, lmin=None, lxmin=None, lxmax=None, lmax=None, lymin=None, lymax=None):
        """ initialize the l_mask library.
                ivf_lib = inverse-variance filter library to wrap.
                
                (optional) lmin  = high-pass multipole filter to apply.
                (optional) lxmin = high-pass multipole filter to apply in the x direction.
                (optional) lymin = high-pass multipole filter to apply in the y direction.
                
                (optional) lmax  = low-pass multipole filter to apply.
                (optional) lxmax = low-pass multipole filter to apply in the x direction.
                (optional) lymax = low-pass multipole filter to apply in the y direction.
        """
        self.ivf_lib = ivf_lib
        self.lmin  = lmin
        self.lmax  = lmax
        self.lxmin = lxmin
        self.lymin = lymin
        self.lxmax = lxmax
        self.lymax = lymax

        super(library_l_mask, self).__init__(ivf_lib.obs_lib)
        self.get_fmask = self.ivf_lib.get_fmask

    def hashdict(self):
        return { 'ivf_lib' : self.ivf_lib.hashdict(),
                 'lmin'    : self.lmin,
                 'lmax'    : self.lmax,
                 'lxmin'   : self.lxmin,
                 'lymin'   : self.lymin,
                 'lxmax'   : self.lxmax,
                 'lymax'   : self.lymax,
                 'super'   : super(library_l_mask, self).hashdict() }

    def get_l_masked(self, ret):
        """ helper function returns a multipole-masked maps.tebfft object. """
        return ret.get_l_masked( lmin=self.lmin, lmax=self.lmax, lxmin=self.lxmin, lxmax=self.lxmax, lymin=self.lymin, lymax=self.lymax )

    def get_fl(self):
        """ return a multipole-masked version of self.ivf_lib.get_fl(). """
        return self.get_l_masked(self.ivf_lib.get_fl())

    def get_sim_teb(self, idx):
        """ return a multipole-masked version of self.ivf_lib.get_sim_teb(idx). """
        return self.get_l_masked( self.ivf_lib.get_sim_teb(idx) )
