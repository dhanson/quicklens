# quicklens/sims/qest.py
# --
# this module contains a library object which applies quadratic estimators
# to inverse-variance filtered sky maps. in this module a 'quadratic estimator'
# is actually a weighted sum of estimates produced by qest.qest. these estimators
# each have a unique key, and are defined in library.init_q.

import os, sys, hashlib
import numpy  as np
import pickle as pk

import quicklens as ql
import util

class library():
    def __init__(self, cl_unl, cl_len, ivfs1, lib_dir=None, ivfs2=None, npad=2):
        """ library for fetching quadratic estimator results, designed to make it easy to combine quadratic estimators.
                cl_unl = spec.camb_clfile object containing unlensed theory power spectra.
                cl_len = spec.camb_clfile object containing lensed theory power spectra.
                ivfs1  = a sims.ivf.library object which which provides the first leg for the quadratic estimators.
                (optional) lib_dir = directory to store hash for this object, as well as cached estimator mean-fields.
                (optional) ivfs2   = a sims.ivf.library object which provides the second leg for the quadratic estimators. defaults to ivf1.
                (optional) npad    = padding to use for convolutions used when computing quadratic estimators.
        """
        
        self.cl_unl     = cl_unl
        self.cl_len     = cl_len
        self.ivfs1      = ivfs1
        self.lib_dir    = lib_dir
        
        if ivfs2 == None:
            ivfs2 = ivfs1
        self.ivfs2      = ivfs2
        
        self.npad       = npad
        
        self.qes        = {} # estimators
        self.qfs        = {} # estimator fields
        self.qrs        = {} # estimator responses

        if lib_dir != None:
            if ql.mpi.rank == 0:
                if not os.path.exists(lib_dir):
                    os.makedirs(lib_dir)

                if not os.path.exists(lib_dir + "/sim_hash.pk"):
                    pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
            ql.mpi.barrier()
            util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def get_fmasks(self):
        """ return the a tupole containing the masks associated with self.ivfs1 and self.ivfs2. """
        return self.ivfs1.get_fmask(), self.ivfs2.get_fmask()

    def hashdict(self):
        """ return a dictionary uniquely associated with the contents of this library. """
        ret = { 'cl_unl' : self.cl_unl.hashdict(),
                'cl_len' : self.cl_len.hashdict(),
                'ivfs1'  : self.ivfs1.hashdict(),
                'npad'   : self.npad }

        if self.ivfs2 is not self.ivfs1:
            ret['ivfs2'] = self.ivfs2.hashdict()

        return ret

    def init_q(self, k):
        """ initialize the estimator with key k.
        this consists of an array of tuples (qe, scaling) containing a qest.qest object
        as well as a scaling to apply (either an overall number, or a 1D array in the
        case of a scale-dependent contribution. the total estimator is then a sum over
        tupoles of each estimator multiplied by its scaling.
        """
        if False:
            pass
        elif k == 'ptt':
            self.qes[k] = [ ( (ql.qest.lens.phi_TT(self.cl_len.cltt), 'TT'), 1.) ]
        elif k == 'pee':
            self.qes[k] = [ ( (ql.qest.lens.phi_EE(self.cl_len.clee), 'EE'), 1.) ]
        elif k == 'pte':
            self.qes[k] = [ ( (ql.qest.lens.phi_TE(self.cl_len.clte), 'TE'), 1.),
                            ( (ql.qest.lens.phi_ET(self.cl_len.clte), 'ET'), 1.) ]
        elif k == 'ptb':
            self.qes[k] = [ ( (ql.qest.lens.phi_TB(self.cl_len.clte), 'TB'), 1.),
                            ( (ql.qest.lens.phi_BT(self.cl_len.clte), 'BT'), 1.) ]
        elif k == 'peb':
            self.qes[k] = [ ( (ql.qest.lens.phi_EB(self.cl_len.clee), 'EB'), 1.),
                            ( (ql.qest.lens.phi_BE(self.cl_len.clee), 'BE'), 1.) ]
        elif k == 'pmv': # minimum-variance lensing estimator.
            self.qes[k] = [ ( 'ptt', 1. ),
                            ( 'pee', 1. ),
                            ( 'pte', 1. ),
                            ( 'ptb', 1. ),
                            ( 'peb', 1. ) ]
        else:
            assert(0)

    def get_qe(self, k):
        """ return an array of tuples (qe, scaling) defining the quadratic estimator k. """
        if k not in self.qes.keys():
            self.init_q(k)
        return self.qes[k]

    def get_qr(self, ke, ks=None):
        """ return a maps.cfft object containing the response of a quadratic estimator 'ke' to another source of statistical anisotropy 'ks'.
                ke = estimator key.
                (optional) ks = source key (defaults to ke).
        """
        if ks == None:
            ks = ke

        if (isinstance(ke, tuple) and isinstance(ks, tuple)): # a (qest.qest, scaling) pair for both estimator and source keys.
            qe, f12, f1, f2 = ke
            qs, f34 = ks

            if f12 == f34:
                return qe.fill_resp( qs, ql.maps.cfft(f1.nx, f1.dx, ny=f1.ny, dy=f1.dy), f1.fft, f2.fft, npad=self.npad)
            else:
                return ql.maps.cfft(f1.nx, f1.dx, ny=f1.ny, dy=f1.dy)

        elif isinstance(ke, tuple): # a (qest.qest, scaling) pair only for the estimator. need to expand the source key.
            qe, f12, f1, f2 = ke
            
            ret = ql.maps.cfft(f1.nx, f1.dx, ny=f1.ny, dy=f1.dy)
            for (tqs, tfs) in self.get_qe(ks):
                ret += self.get_qr( ke, ks=tqs) * tfs
            return ret
            
        else: # keys for both the estimator and source. need to expand.
            if (ke,ks) not in self.qrs.keys():
                tfl1, efl1, bfl1 = self.ivfs1.get_fl().get_cffts()
                if self.ivfs2 is not self.ivfs1:
                    tfl2, efl2, bfl2 = self.ivfs2.get_fl().get_cffts()
                else:
                    tfl2, efl2, bfl2 = tfl1, efl1, bfl1
                
                ret = ql.maps.cfft(tfl1.nx, tfl1.dx, ny=tfl1.ny, dy=tfl1.dy)
                for tqe, tfe in self.get_qe(ke):
                    if not isinstance(tqe, tuple):
                        ret += self.get_qr( tqe, ks=ks ) * tfe
                    else:
                        qe, f12 = tqe
            
                        f1 = {'T' : tfl1, 'E' : efl1, 'B' : bfl1}[f12[0]]
                        f2 = {'T' : tfl2, 'E' : efl2, 'B' : bfl2}[f12[1]]
                        
                        for (tqs, tfs) in self.get_qe(ks):
                            ret += self.get_qr( (qe,f12,f1,f2), ks=tqs ) * tfe * tfs
                self.qrs[(ke,ks)] = ret
            return self.qrs[(ke,ks)]

    def get_qft(self, k, tft1, eft1, bft1, tft2, eft2, bft2):
        """ return the estimate for key k.
                k          = estimator key.
                [t,e,b]ft1 = maps.cfft objects containing the inverse-variance
                             filtered Fourier modes for the first leg in the estimator.
                             should be taken from self.ivfs1.
                [t,e,b]ft2 = maps.cfft objects containing the inverse-variance
                             filtered Fourier modes for the first leg in the estimator.
                             should be taken from self.ivfs2.
        """
        
        ret = ql.maps.cfft(tft1.nx, tft1.dx, ny=tft1.ny, dy=tft1.dy)
        for tqe, tfe in self.get_qe(k):
            if not isinstance(tqe, tuple):
                ret += self.get_qft( tqe, tft1, eft1, bft1, tft2, eft2, bft2 ) * tfe
            else:
                (qe, f12) = tqe

                f1 = {'T' : tft1, 'E' : eft1, 'B' : bft1}[f12[0]]
                f2 = {'T' : tft2, 'E' : eft2, 'B' : bft2}[f12[1]]
                ret += qe.eval(f1, f2, npad=self.npad) * tfe
        return ret

    def get_sim_tebfts(self, i):
        """ return a 6-tuple containing maps.cfft objects for t, e, b for the first and second legs of the estimator for simulation 'i'. """
        tft1, eft1, bft1 = self.ivfs1.get_sim_teb(i).get_cffts()
        if self.ivfs2 is not self.ivfs1:
            tft2, eft2, bft2 = self.ivfs2.get_sim_teb(i).get_cffts()
        else:
            tft2, eft2, bft2 = tft1, eft1, bft1
        return tft1, eft1, bft1, tft2, eft2, bft2

    def get_sim_qft(self, k, i):
        """ return the quadratic estimator for key 'k' on simulation 'i'. note that this estimator is unnormalized. """
        tft1, eft1, bft1, tft2, eft2, bft2 = self.get_sim_tebfts(i)
        return self.get_qft(k, tft1, eft1, bft1, tft2, eft2, bft2)

    def get_sim_qft_mf(self, k, idxs):
        """ return the mean-field for quadratic estimator with key 'k', averaged over indexes in the array 'idxs'. not that this mean-field is unnormalized. """
        tfname = self.lib_dir.format(prefix="temp") + "/sim_qft_mf_%s_%s.pk" % (k, hashlib.sha1(idxs).hexdigest())
        if not os.path.exists(tfname):
            qft_mf_avg = ql.util.avg()
            for i, idx in ql.util.enumerate_progress(idxs, "sims.qest.library.get_sim_qft_mf"):
                qft_mf_avg.add( self.get_sim_qft(k, idx) )
            pk.dump( qft_mf_avg.get(), open(tfname, 'w') )
        return pk.load( open(tfname, 'r') )

class library_kappa():
    """ a helper library which takes input kappa maps from a cmb simulation library and returns them as though they were the output of a quadratic estimator. """
    def __init__(self, lib_qest, cmbs):
        """ initialize the library.
                lib_qest = a sims.qest.library object to mimic.
                cmbs     = library with a get_sim_kappa(i) method which returns a maps.rmap object containing lensing kappa for simulation i.
        """
        self.lib_qest = lib_qest
        self.cmbs     = cmbs

        assert(hasattr(cmbs, 'get_sim_kappa'))
        self.get_qr   = self.lib_qest.get_qr

        m1, m2 = self.lib_qest.ivfs1.get_fmask(), self.lib_qest.ivfs2.get_fmask()
        self.mask1 = np.ones(m1.shape)
        self.mask2 = np.ones(m2.shape)

    def hashdict(self):
        return { 'lib_qest' : self.lib_qest.hashdict(),
                 'cmbs'     : self.cmbs.hashdict() }

    def get_fmasks(self):
        """ return effective mask for kappa. """
        return self.mask1, self.mask2

    def get_sim_qft(self, k, i):
        """ return a mimic of self.lib_qest.get_sim_qft(k,i) containing the true lensing kappa map. """
        return self.cmbs.get_sim_kappa(i).get_cfft() * np.nan_to_num( 2./np.arange(0., 100000.)**2 ) * self.get_qr(k) # convert kappa->phi, then convolve to look like qest.

    def get_sim_qft_mf(self, k, idxs):
        """ there is no mean-field for this library, so always returns 0. """
        assert( len(idxs) == 0 ) # sanity-check, can be removed if desired.
        return 0.
