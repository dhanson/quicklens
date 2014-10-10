# quicklens/sims/obs.py
# --
# this module contains code for simulating observations of sky maps.

import os, sys, hashlib
import numpy  as np
import pickle as pk

import quicklens as ql
import phas
import util

class library_white_noise():
    """ a library which contains simulated Gaussian noise which is uncorrelated between pixels (and will therefore have a white power spectrum.) """
    def __init__(self, pix, transf, sky_lib, lib_dir, nlev_t=0.0, nlev_p=0.0, phase=None):
        """ initialize the library.
             pix     = a maps.pix object with the sky pixelization.
             transf  = a transfer function (with a multiplication operation defined for maps.tebbft),
                       usually describing the effect of an instrumental beam.
             sky_lib = a library object with a get_sim_tqu() method which provides the sky map realizations.
             lib_dir = directory to store the hash for the library.
             nlev_t  = temperature noise level. either a scalar or an map-sized 2D array.
                       noise realizations in each temperature pixel are given by nlev_t * standard normal.
             nlev_p  = polarization noise level. either a scalar or an map-sized 2D array.
                       noise realizations in each polarization pixel are given by nlev_p * standard normal.
             (optional) phase = phas.library object used to control the random number stream.
        """
        self.pix     = pix
        self.transf  = transf
        self.sky_lib = sky_lib
        self.lib_dir = lib_dir
        self.phase   = phase

        self.nlev_t  = nlev_t
        self.nlev_p  = nlev_p

        if (self.phase == None):
            self.phase = phas.library( 3*pix.nx*pix.ny, lib_dir = self.lib_dir + "/phase" )

        if (ql.mpi.rank == 0):
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(lib_dir + "/sim_hash.pk"):
                pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
        ql.mpi.barrier()
        util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def hashdict(self):
        ret = { 'pix'     : self.pix.hashdict(),
                'sky_lib' : self.sky_lib.hashdict(),
                'phase'   : self.phase.hashdict() }

        if type(self.transf) == np.ndarray:
            ret['transf'] = hashlib.sha1(self.transf.view(np.uint8)).hexdigest()
        else:
            ret['transf'] = self.transf.hashdict()

        if np.isscalar(self.nlev_t):
            ret['nlev_t'] = self.nlev_t
        else:
            ret['nlev_t'] = self.nlev_t.hashdict()

        if np.isscalar(self.nlev_p):
            ret['nlev_p'] = self.nlev_p
        else:
            ret['nlev_p'] = self.nlev_p.hashdict()

        return ret

    def get_sim_tqu(self, idx):
        # fetch the sky simulation
        sky_tqu = self.sky_lib.get_sim_tqu(idx)
        sky_teb = sky_tqu.get_teb()

        # convolve with transfer function
        sky_teb = sky_teb * self.transf

        # generate the noise component
        nx, dx, ny, dy = sky_teb.nx, sky_teb.dx, sky_teb.ny, sky_teb.dy

        nmap = ql.maps.tqumap( nx, dx, ny=ny, dy=dy )
        self.phase.set_state(idx)
        nmap.tmap = np.random.standard_normal(sky_tqu.tmap.shape) * self.nlev_t / (180.*60./np.pi*np.sqrt(dx * dy))
        nmap.qmap = np.random.standard_normal(sky_tqu.qmap.shape) * self.nlev_p / (180.*60./np.pi*np.sqrt(dx * dy))
        nmap.umap = np.random.standard_normal(sky_tqu.umap.shape) * self.nlev_p / (180.*60./np.pi*np.sqrt(dx * dy))
        self.phase.check_state_final(idx)

        # sum and return
        return sky_teb.get_tqu() + nmap
