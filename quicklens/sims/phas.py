# quicklens/sims/phas.py
# --
# this module contains code for maintaining the state of the numpy random number generator,
# use in Monte Carlo simulations. the phas.library() class can be used for making reproducible
# random arrays of a given size, indexed by a simulation number 'idx'.
# typical sizes for a simulation are
# 
#  * alm (using healpy.synalm)              = n * ( (lmax+1)*(lmax+2) ), where n = # of spectra
#  * real Fourier modes (maps.rfft)         = 2 * nx*(ny/2+1)
#  * real T,E,B Fourier modes (maps.tebfft) = 6 * nx*(ny/2+1)
# 

import os
import numpy as np
import pickle as pk

from .. import mpi
import util

class library():
    """ library for maintaining the state of the numpy random number generator. """
    def __init__(self, size, lib_dir, seed=None, random=np.random.standard_normal):
        """ initialize the library.
             * size              = size of array to be simulated for for each index.
             * lib_dir           = directory to store information on the random state for each index.
             * (optional) seed   = np.random seed for the 0th index. if set to None, then seed will
                                   be initialized from clock after prompting user to enter a keystroke.
             * (optional) random = function random(size) for calling the random number generator.
        """
        self.size    = size
        self.lib_dir = lib_dir
        self.random  = random

        if mpi.rank == 0:
            if not os.path.exists(lib_dir):
                os.makedirs(lib_dir)

            if not os.path.exists(lib_dir + "/state_%04d.pk"%0):
                # intialize seed.
                if seed != None:
                    np.random.seed(seed)
                else:
                    keyseed = raw_input("quicklens::sims::phas: enter several random strokes on the keyboard followed by enter to initialize the random seed.\n")
                    assert(len(keyseed) > 0)
                    #Ensure that seed is less than or equal to 2**32
                    np.random.seed(np.abs(int(hash(keyseed)))%(2**32))

                # store the current state.
                pk.dump( np.random.get_state(), open(lib_dir + "/state_%04d.pk"%0, 'w') )

            if not os.path.exists(lib_dir + "/sim_hash.pk"):
                pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )

        mpi.barrier()
        util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

        if seed != None:
            np.random.seed(seed)
            self.random(size=self.size)
            self.check_state_final(0)

    def hashdict(self):
        """ returns a dictionary which uniquely characterizes this object. """
        return { 'size' : self.size,
                 'init' : dict( zip( np.arange(0,4), pk.load( open(self.lib_dir + "/state_%04d.pk"%0, 'r') ) ) ) }

    def set_state(self, idx):
        """ set the random number generator to the state used for simulation with index 'idx'. """
        np.random.set_state( self.get_state(idx) )

    def get_state(self, idx):
        """ return the np.random state for the simulation with index 'idx'. """
        if not os.path.exists(self.lib_dir + "/state_%04d.pk"%idx):
            assert(mpi.rank == 0)
            state = self.get_state(idx-1)
            np.random.set_state(state)
            print "quicklens::sims::phas: caching state %d"%idx
            self.random(size=self.size)
            pk.dump( np.random.get_state(), open(self.lib_dir + "/state_%04d.pk"%idx, 'w') )
        return pk.load( open(self.lib_dir + "/state_%04d.pk"%idx, 'r') )

    def check_state_final(self, idx):
        """ after generating the random numbers for the state with index 'idx',
        check that the state of the random number is equal to the state for index idx+1. """
        fstate = np.random.get_state()
        mstate = self.get_state(idx+1)
        assert( np.all( [ np.all(f == m) for f,m in zip(fstate, mstate) ] ) )

class library_cut(library):
    """ a wrapper library around phas.library, which truncates each simulation array to a smaller length. """
    def __init__(self, size, flib):
        self.size = size
        self.flib = flib

        assert( self.size <= self.flib.size )
        
    def hashdict(self):
        return {'size' : self.size,
                'flib' : self.flib.hashdict() }

    def get_state(self, idx):
        return self.flib.get_state(idx)

    def check_state_final(self,idx):
        self.flib.random(size=(self.flib.size - self.size))
        fstate = np.random.get_state()
        mstate = self.get_state(idx+1)
        assert( np.all( [ np.all(f == m) for f,m in zip(fstate, mstate) ] ) )
