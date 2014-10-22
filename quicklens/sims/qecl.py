# quicklens/sims/qecl.py
# --
# this module contains library objects for taking the auto- and cross-spectra
# of the quadratic estimator libraries in sims/qest.py. it also contains code
# for calculation of N0 biases.
#

import os, sys, hashlib
import numpy  as np
import pickle as pk

import quicklens as ql
import util

class library():
    """ a library for taking the auto- or cross-spectrum of the quadratic estimator
    libraries in sims/qest.py """
    def __init__(self, qeA, lib_dir, qeB=None, mc_sims_mf=None, npad=2, lmax_lcl=5000, maxcache=True):
        """ initialize the qecl library.
                qeA     = sims.qest.library object providing the first quadratic estimator.
                lib_dir = directory to store the hash of this object, as well as cached spectra.
                (optional) qeB        =
                (optional) mc_sims_mf = simulations to use when calculating the mean-fields.
                                        to avoid monte carlo noise bias, different sims are always
                                        used for mfA and mfB. these can be specified explicitly by
                                        passing a tuple (mc_sims_mfA, mc_sims_mfB). if a single array
                                        is passed then instead (mc_sims_mf[::2], mc_sims_mf[1::2]) is
                                        used. if mc_sims_mf=None then no mean-field subtraction is
                                        performed.
                (optional) npad       = padding factor to apply when calculating the semi-analytical N0 bias.
                (optional) maxcache   = aggressively cache to disk when calculating N0 biases.
        """
        if not qeB:
            qeB = qeA

        self.qeA         = qeA
        self.qeB         = qeB
        self.npad        = npad
        self.lmax_lcl    = lmax_lcl
        self.maxcache    = maxcache

        self.lib_dir     = lib_dir

        if isinstance(mc_sims_mf, tuple):
            self.mc_sims_mfA, self.mc_sims_mfB = mc_sims_mf
        else:
            if mc_sims_mf == None:
                self.mc_sims_mfA = None
                self.mc_sims_mfB = None
            else:
                self.mc_sims_mfA = mc_sims_mf[0::2].copy()
                self.mc_sims_mfB = mc_sims_mf[1::2].copy()
        
        if ql.mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(lib_dir + "/sim_hash.pk"):
                pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )

            # calculate several heuristics for the sky overlap between
            # qeA.ivfs and qeB.ivfs, to be used for correcting the spectra.
            if not os.path.exists(lib_dir + "/qecl_fcut.dat"):
                mask1, mask2 = [ m.flatten() for m in self.qeA.get_fmasks() ]
                mask3, mask4 = [ m.flatten() for m in self.qeB.get_fmasks() ]

                shape = mask1.shape
                assert( shape == mask2.shape )
                assert( shape == mask3.shape )
                assert( shape == mask4.shape )
                npix = mask1.size

                fcut11 = np.sum( mask1**2 ) / npix
                fcut12 = np.sum( mask1 * mask2 ) / npix
                fcut13 = np.sum( mask1 * mask3 ) / npix
                fcut14 = np.sum( mask1 * mask4 ) / npix
                fcut22 = np.sum( mask2**2 ) / npix
                fcut23 = np.sum( mask2 * mask3 ) / npix
                fcut24 = np.sum( mask2 * mask4 ) / npix
                fcut33 = np.sum( mask3**2 ) / npix
                fcut34 = np.sum( mask3 * mask4 ) / npix
                fcut44 = np.sum( mask4**2 ) / npix

                fcut1234 = np.sum( mask1 * mask2 * mask3 * mask4 ) / npix

                np.savetxt( lib_dir.format(prefix="outputs") + "/qecl_fcut.dat",
                            [ fcut1234,
                              fcut11, fcut12, fcut13, fcut14,
                              fcut22, fcut23, fcut24,
                              fcut33, fcut34,
                              fcut44 ] )
        ql.mpi.barrier()
        util.hash_check( pk.load( open(lib_dir.format(prefix="outputs") + "/sim_hash.pk", 'r') ), self.hashdict() )

        [ self.fcut1234,
          self.fcut11, self.fcut12, self.fcut13, self.fcut14,
          self.fcut22, self.fcut23, self.fcut24,
          self.fcut33, self.fcut34,
          self.fcut44 ] = np.loadtxt(lib_dir.format(prefix="outputs") + "/qecl_fcut.dat")

    def hashdict(self):
        """ return a dictionary uniquely associated with the contents of this library. """
        ret = { 'qeA'        : self.qeA.hashdict(),
                'mc_sims_mfA': self.mc_sims_mfA,
                'mc_sims_mfB': self.mc_sims_mfB,
                'npad'       : self.npad,
                'lmax_lcl'   : self.lmax_lcl }
        
        if self.qeB is not self.qeA:
            ret['qeB'] = self.qeB.hashdict()

        return ret

    def get_qcr(self, kA, sA=None, kB=None, sB=None):
        """ return a maps.cfft object containing the response of an estimator cross-spectrum (qeA x qeB) to a source of statistical anisotrpy.
                k1 = estimator key for qeA.
                (optional) sA = key defining source of anisotropy for the qeA estimator. defaults to kA.
                (optional) kB = estimator key for qeB. defaults to kA.
                (optional) sB = key defining source of anisotropy for the qeB estimator. defaults to kB.
        """
        if sA == None:
            sA = kA
        if kB == None:
            kB = kA
        if sB == None:
            sB = kB

        return self.qeA.get_qr(kA, sA) * self.qeB.get_qr(kB, sB)

    def get_qcr_lcl(self, *args, **kwargs):
        """ caching version of get_qcr, using a spec.lcl object to compress the full 2D response. """
        tfname = self.lib_dir + ("/cache_lcl_%+d_%s.pk" % (hash(args + tuple(frozenset(kwargs.items()))), "get_qcr"))
        tfname = tfname.replace('+', 'p').replace('-', 'm')
        if not os.path.exists(tfname):
            print "caching lm: %s" % tfname
            lm = ql.spec.lcl( self.lmax_lcl, self.get_qcr(*args, **kwargs) ) 
            pk.dump( lm, open(tfname,'w') )
        return pk.load( open(tfname,'r') )
    
    def get_sim_qcl(self, kA, i, kB=None):
        """ return a maps.cfft object containing the cross-power between two quadratic estimators (qeA x qeB).
                kA = estimator key for qeA.
                i  = simulation index.
                (optional) kB = estimator key for qeB (defaults to kA).

                the cross-spectrum is corrected for the sky cut.
        """
        if kB == None:
            kB = kA

        # obtain the quadratic estimates.
        qeA_qft = self.qeA.get_sim_qft(kA, i)
        qeB_qft = self.qeB.get_sim_qft(kB, i)

        # perform mean field subtraction.
        if self.mc_sims_mfA != None:
            assert( idx not in self.mc_sims_mfA )
            qeA_qft -= self.qeA.get_sim_qft_mf(k1, self.mc_sims_mfA)

        if self.mc_sims_mfB != None:
            assert( idx not in self.mc_sims_mfB )
            qeB_qft -= self.qeB.get_sim_qft_mf(k2, self.mc_sims_mfB)

        # return the cross-power, after correcting for sky cut.
        return ql.maps.cfft( qeA_qft.nx, qeA_qft.dx, fft=(qeA_qft.fft * np.conj(qeB_qft.fft)) / self.fcut1234, ny=qeA_qft.ny, dy=qeB_qft.dy )

    def get_sim_qcl_lcl(self, *args, **kwargs):
        """ caching version of get_sim_qcl, using a spec.lcl object to compress the full 2D response. """
        tfname = self.lib_dir + ("/cache_lcl_%+d_%s.pk" % (hash(args + tuple(frozenset(kwargs.items()))), "get_sim_qcl"))
        tfname = tfname.replace('+', 'p').replace('-', 'm')
        if not os.path.exists(tfname):
            print "caching lm: %s" % tfname
            lm = ql.spec.lcl( self.lmax_lcl, self.get_sim_qcl(*args, **kwargs) )
            pk.dump( lm, open(tfname,'w') )
        return pk.load( open(tfname,'r') )

    def get_ncl_helper(self, kA, kB,
                       t1, e1, b1, t2, e2, b2,
                       t3, e3, b3, t4, e4, b4):

        if (isinstance(kA, tuple) and isinstance(kB, tuple)):
            q12, f12 = kA
            q34, f34 = kB

            f1, f2  = f12
            f1 = {'T' : t1, 'E' : e1, 'B' : b1}[f1]
            f2 = {'T' : t2, 'E' : e2, 'B' : b2}[f2]

            f3, f4 = f34
            f3 = {'T' : t3, 'E' : e3, 'B' : b3}[f3]
            f4 = {'T' : t4, 'E' : e4, 'B' : b4}[f4]
            
            cfft = ql.maps.cfft( nx=t1.nx, dx=t1.dx, ny=t1.ny, dy=t1.dy )
            ql.qest.qe_cov_fill_helper( q12, q34, cfft,
                                        f1.fft*np.conj(f3.fft)/self.fcut13, f2.fft*np.conj(f4.fft)/self.fcut24, switch_ZA=False, conj_ZA=True, npad=self.npad )
            ql.qest.qe_cov_fill_helper( q12, q34, cfft,
                                        f1.fft*np.conj(f4.fft)/self.fcut14, f2.fft*np.conj(f3.fft)/self.fcut23, switch_ZA=True,  conj_ZA=True, npad=self.npad )

            return ql.spec.lcl( self.lmax_lcl, cfft )
        elif isinstance(kA, tuple):
            lm = ql.util.sum()
            for (tq34, tf34) in self.qeB.get_qe(kB):
                lm.add( self.get_ncl_helper( kA, tq34, t1, e1, b1, t2, e2, b2, t3, e3, b3, t4, e4, b4) * tf34 )
            return lm.get()
        else:
            lm = ql.util.sum()
            for (tq12, tf12) in self.qeA.get_qe(kA):
                if not isinstance(tq12, tuple):
                    lm.add( self.get_ncl_helper(tq12, kB, t1, e1, b1, t2, e2, b2, t3, e3, b3, t4, e4, b4) * tf12 )
                else:
                    for (tq34, tf34) in self.qeB.get_qe(kB):
                        lm.add( self.get_ncl_helper( tq12, tq34, t1, e1, b1, t2, e2, b2, t3, e3, b3, t4, e4, b4 ) * tf12 * tf34 )
        return lm.get()
    
    def get_sim_ncl_lcl(self, kA, i, kB=None):
        """ return a spec.lcl object containing a semi-analytic estimate
        of the N0 bias for the cross-power between two quadratic
        estimators (qeA x qeB). the bias is calculated using the
        2D cross-power for simulation i, treating the Fourier modes as
        uncorrelated.
        
                kA = estimator key for qeA.
                i  = simulation index.
                (optional) kB = estimator key for qeB (defaults to kA).
        """
        if kB == None:
            kB = kA

        assert( not isinstance(kA, tuple) )
        assert( not isinstance(kB, tuple) )
            
        if self.qeA is self.qeB:
            kA, kB = sorted([kA,kB])

        tfname = self.lib_dir + ("/cache_lcl_%+d_%s.pk" % (hash( (kA,i) + tuple(frozenset({'kB' : kB}.items()))), "get_sim_ncl_lcl"))
        tfname = tfname.replace('+', 'p').replace('-', 'm')
                
        if not os.path.exists(tfname):
            print "get_sim_ncl_lcl caching lm for kA=%s, kB=%s, i=%d : %s" % (kA, kB, i, tfname)

            @ql.util.memoize
            def get_tebfts(i):
                # fetches the inverse-variance filtered FFTs for sim i.
                # wrapped with memoize decorator to reduce # of calls.
                t1, e1, b1, t2, e2, b2 = self.qeA.get_sim_tebfts(i)
                if self.qeA is not self.qeB:
                    t3, e3, b3, t4, e4, b4 = self.qeB.get_sim_tebfts(i)
                else:
                    t3, e3, b3, t4, e4, b4 = t1, e1, b1, t2, e2, b2
                return t1, e1, b1, t2, e2, b3, t3, e3, b3, t4, e4, b4

            if self.maxcache == True:
                # implementation 1 -- maximal caching for composite estimators.
                lm = ql.util.sum()
                for (tq12, tf12) in self.qeA.get_qe(kA):
                    if not isinstance(tq12, tuple):
                        lm.add( self.get_sim_ncl_lcl(tq12, i, kB=kB) * tf12 )
                    else:
                        for (tq34, tf34) in self.qeB.get_qe(kB):
                            if not isinstance(tq34, tuple):
                                lm.add( self.get_ncl_helper( tq12, tq34, *get_tebfts(i) ) * tf12 * tf34 )
                            else:
                                lm.add( self.get_ncl_helper( tq12, tq34, *get_tebfts(i) ) * tf12 * tf34 )
                ret = lm.get()
            else:
                # implementation 2 -- minimal caching for composite estimators.
                ret = self.get_ncl_helper( kA, kB, *get_tebfts(i) )
            
            pk.dump( ret, open(tfname,'w') )

        return pk.load( open(tfname,'r') )
