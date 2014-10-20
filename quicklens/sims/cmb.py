# quicklens/sims/cmb.py
# --
# this module contains code for simulating lensed/unlensed flat-sky cmb maps.

import os
import numpy as np
import pickle as pk

import sims
import phas
import util

import quicklens as ql

class library_flat_unlensed():
    """ a library which generates Gaussian CMB realizations. """
    def __init__(self, pix, cl_unl, lib_dir, phase=None):
        """ initialize the library.
             pix     = a maps.pix object with the sky pixelization.
             cl_unl  = a spec.camb_clfile object containing the CMB temperature+polarization power spectra.
             lib_dir = directory to store the hash for the library.
             (optional) phase = phas.library object used to control the random number stream.
        """
        self.pix     = pix
        self.cl_unl  = cl_unl
        self.lib_dir = lib_dir
        self.phase   = phase

        if (self.phase == None):
            lmax = cl_unl.lmax
            self.phase = phas.library( 8*pix.nx*(pix.ny/2+1), lib_dir = self.lib_dir + "/phase" )

        if (ql.mpi.rank == 0):
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(lib_dir + "/sim_hash.pk"):
                pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
        ql.mpi.barrier()
        util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def hashdict(self):
        return { 'pix'    : self.pix.hashdict(),
                 'cl_unl' : self.cl_unl.hashdict(),
                 'phase'  : self.phase.hashdict() }

    def get_sim_tqu(self, idx):
        self.phase.set_state(idx)
        tebfft = ql.sims.tebfft( self.pix, self.cl_unl )
        phifft = ql.sims.rfft( self.pix, self.cl_unl.cltt )
        self.phase.check_state_final(idx)

        return tebfft.get_tqu()

class library_flat_lensed():
    """ a library which generates lensed CMB realizations. """
    def __init__(self, pix, cl_unl, lib_dir, phase=None):
        """ initialize the library.
             pix     = a maps.pix object with the sky pixelization.
             cl_unl  = a spec.camb_clfile object containing the unlensed CMB temperature+polarization power spectra.
             lib_dir = directory to store the hash for the library.
             (optional) phase = phas.library object used to control the random number stream.
        """
        self.pix     = pix
        self.cl_unl  = cl_unl
        self.lib_dir = lib_dir
        self.phase   = phase

        if self.phase == None:
            lmax = cl_unl.lmax
            self.phase = phas.library( 8*pix.nx*(pix.ny/2+1), lib_dir = self.lib_dir + "/phase" )

        if ql.mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(lib_dir + "/sim_hash.pk"):
                pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
        ql.mpi.barrier()
        util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def hashdict(self):
        return { 'pix'    : self.pix.hashdict(),
                 'cl_unl' : self.cl_unl.hashdict(),
                 'phase'  : self.phase.hashdict() }

    def cache_sim(self, idx):
        tfname_phi_fft = self.lib_dir + "/sim_" + ('%04d' % idx) + "_phi_fft.pk"
        tfname_teb_unl = self.lib_dir + "/sim_" + ('%04d' % idx) + "_teb_unl.pk"
        tfname_tqu_len = self.lib_dir + "/sim_" + ('%04d' % idx) + "_tqu_len.pk"

        assert( not os.path.exists(tfname_phi_fft) )
        assert( not os.path.exists(tfname_teb_unl) )
        assert( not os.path.exists(tfname_tqu_len) )

        self.phase.set_state(idx)
        teb_unl = sims.tebfft( self.pix, self.cl_unl )
        phi_fft = sims.rfft( self.pix, self.cl_unl.clpp )
        self.phase.check_state_final(idx)
        
        tqu_unl = teb_unl.get_tqu()
        tqu_len = ql.lens.make_lensed_map_flat_sky( tqu_unl, phi_fft )

        assert( not os.path.exists(tfname_phi_fft) )
        assert( not os.path.exists(tfname_teb_unl) )
        assert( not os.path.exists(tfname_tqu_len) )

        pk.dump( phi_fft, open(tfname_phi_fft, 'w') )
        pk.dump( teb_unl, open(tfname_teb_unl, 'w') )
        pk.dump( tqu_len, open(tfname_tqu_len, 'w') )

    def get_sim_phi(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_phi_fft.pk"
        if not os.path.exists(tfname):
            self.cache_sim(idx)
        return pk.load( open(tfname, 'r' ) ).get_rmap()

    def get_sim_kappa(self, idx, fl=1.):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_phi_fft.pk"
        if not os.path.exists(tfname):
            self.cache_sim(idx)
        return ( ( pk.load( open(tfname, 'r' ) ) * ( 0.5 * np.arange(0., 10000.)**2 ) ) * fl ).get_rmap()

    def get_sim_tqu(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_tqu_len.pk"
        if not os.path.exists(tfname):
            self.cache_sim(idx)
        return pk.load( open(tfname,'r') )

    def get_sim_phase(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_teb_unl.pk"
        if not os.path.exists(tfname):
            self.cache_sim(idx)
        return ql.spec.clmat_teb( self.cl_unl ).cholesky().inverse() * pk.load( open(tfname,'r') )
