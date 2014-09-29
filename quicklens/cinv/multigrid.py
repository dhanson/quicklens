# quicklens/cinv/multigrid.py
# --
# this module contains tools for using a multigrid preconditioner in
# C^{-1} filtering, following http://arxiv.org/abs/0705.3980
# it is meant to be used in conjunction with the functions
# and filtering classes found in quicklens/cinv/opfilt_teb.py
#
# an example of the multigrid code in use can be found in
# examples/cinv/test_cinv_teb.py, with the multigrid=True option enabled.
#

import sys, os, re, glob, copy
import numpy as np

import cd_solve
import cd_monitors
from .. import util
from .. import maps

# ===

class multigrid_stage(object):
    """ a resolution stage in the multigrid preconditioner. """
    def __init__(self, id, pix, iter_max, eps_min, tr, cache):
        self.depth         = id
        self.pix           = pix
        self.iter_max      = iter_max
        self.eps_min       = eps_min
        self.tr            = tr
        self.cache         = cache
        self.pre_ops       = [] 

class chain():
    """ a chain object constructs a series of multigrid resolution stages. """
    def __init__(self, nstages, opfilt, s_inv_filt, n_inv_filt, plogdepth=0, eps_min=1.e-6, stage_iter_max=3):
        """ initialize this multigrid chain by constructing the resolution stages.
             * nstages        = total number of stages in the chain.
             * opfilt         = module defining dot_op, fwd_op, calc_prep, calc_fini, and pre_op_diag filter functions.
             * s_inv_filt     = the S^{-1} filter for the highest resolution stage in the chain.
             * n_inv_filt     = the N^{-1} filter for the highest resolution stage in the chain.
             * plogdepth      = the lowest stage at which to print convergence information.
             * eps_min        = the convergence criterion.
             * stage_iter_max = the maximum number of iterations in the substages.
        """
        self.opfilt         = opfilt

        self.s_inv_filt     = s_inv_filt
        self.n_inv_filt     = n_inv_filt

        self.plogdepth      = plogdepth
        self.eps_min        = eps_min
        self.stage_iter_max = stage_iter_max

        self.iter_max       = np.inf

        pre_op              = opfilt.pre_op_diag( s_inv_filt, n_inv_filt.degrade(2**nstages) )

        for i in xrange(0, nstages):
            class slog(object):
                def __init__(self, id, cobj):
                    self.id = 1*id
                    self.cobj = cobj
                def log(self, iter, eps, **kwargs):
                    self.cobj.log( self.id, iter, eps, **kwargs )

            pre_op = pre_op_split( pre_op_multigrid(opfilt, s_inv_filt, n_inv_filt.degrade(2**(nstages-i)),
                                                    [pre_op], slog(nstages-i, self).log, cd_solve.tr_cg,
                                                    cd_solve.cache_mem(), stage_iter_max, 0.0 ),
                                   opfilt.pre_op_diag( s_inv_filt, n_inv_filt.degrade(2**(nstages-i-1)) ) )
            
        self.pre_op = pre_op

    def solve( self, x, tqu ):
        self.watch = util.stopwatch()

        self.iter_tot   = 0
        self.prev_eps   = None

        logger = (lambda iter, eps, stage=0, **kwargs :
                  self.log(stage, iter, eps, **kwargs))

        monitor = cd_monitors.monitor_basic(self.opfilt.dot_op(), logger=logger, iter_max=self.iter_max, eps_min=self.eps_min)

        cd_solve.cd_solve( x = x,
                           b = self.opfilt.calc_prep(tqu, self.s_inv_filt, self.n_inv_filt),
                           fwd_op = self.opfilt.fwd_op(self.s_inv_filt, self.n_inv_filt),
                           pre_ops = [self.pre_op], dot_op = self.opfilt.dot_op(),
                           criterion = monitor, tr=cd_solve.tr_cg, cache=cd_solve.cache_mem() )

        return self.opfilt.calc_fini( x, self.s_inv_filt, self.n_inv_filt)

    def log(self, stage, iter, eps, **kwargs):
        self.iter_tot += 1
        elapsed = self.watch.elapsed()

        if stage > self.plogdepth:
            return

        log_str = ('   ')*stage + '[%s] (%d, %f)' % (str(elapsed), iter, eps) + '\n'
        sys.stdout.write(log_str)

class pre_op_split():
    """ a split preconditioner, which splices together the low resolution preconditioner pre_op_low and pre_op_hgh. """
    def __init__(self, pre_op_low, pre_op_hgh):
        self.pre_op_low = pre_op_low
        self.pre_op_hgh = pre_op_hgh

        self.iter   = 0

    def __call__(self, teb):
        return self.calc(teb)

    def calc(self, teb):
        self.iter += 1

        teb_low = self.pre_op_low(teb.degrade(2))
        teb_hgh = self.pre_op_hgh(teb)

        teb_hgh.tfft[0:teb_low.nx,0:teb_low.ny/2+1] = teb_low.tfft
        teb_hgh.efft[0:teb_low.nx,0:teb_low.ny/2+1] = teb_low.efft
        teb_hgh.bfft[0:teb_low.nx,0:teb_low.ny/2+1] = teb_low.bfft
        
        return teb_hgh

class pre_op_multigrid():
    def __init__(self, opfilt, s_inv_filt, n_inv_filt, pre_ops,
                 logger, tr, cache, iter_max, eps_min ):

        self.opfilt   = opfilt
        self.fwd_op   = opfilt.fwd_op(s_inv_filt, n_inv_filt)

        self.pre_ops  = pre_ops

        self.logger   = logger

        self.tr       = tr
        self.cache    = cache

        self.iter_max = iter_max
        self.eps_min  = eps_min

    def __call__(self, talm):
        return self.calc(talm)

    def calc(self, teb):
        monitor = cd_monitors.monitor_basic(self.opfilt.dot_op(), iter_max=self.iter_max, eps_min=self.eps_min, logger=self.logger)

        x = maps.tebfft( teb.nx, teb.dx, ny=teb.ny, dy=teb.dy )

        cd_solve.cd_solve( x, teb,
                           self.fwd_op, self.pre_ops, self.opfilt.dot_op(),
                           monitor, tr=self.tr, cache=self.cache )

        return x
