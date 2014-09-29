# quicklens/cinv/cd_monitors.py
# --
# classes and routines for monitoring/logging the conjugate directions solver in cd_solve.py.
#  monitors = object which is called at each descent iteration, responsible for deciding when converged and also for coordinating any logging/status updates.
#  logger   = function which are called on by a descent monitor to print or log status updates.

import sys
import numpy as np

from .. import util

logger_basic = (lambda it, eps, watch=None, **kwargs : sys.stdout.write( '[' + str(watch.elapsed()) + '] ' + str((it, eps)) + '\n' ))
logger_none  = (lambda it, eps, watch=None, **kwargs : 0)

class monitor_basic():
    """ a simple class for monitoring a conjugate descent iteration. """
    def __init__(self, dot_op, iter_max=np.inf, eps_min=1.0e-10, logger=logger_basic):
        self.dot_op   = dot_op
        self.iter_max = iter_max
        self.eps_min  = eps_min
        self.logger   = logger

        self.watch = util.stopwatch()

    def criterion(self, it, soltn, resid):
        delta = self.dot_op( resid, resid )
        
        if (it == 0):
            self.d0 = delta

        if (self.logger is not None): self.logger( it, np.sqrt(delta/self.d0), watch=self.watch,
                                                   soltn=soltn, resid=resid )

        if (it >= self.iter_max) or (delta <= self.eps_min**2 * self.d0):
            return True

        return False

    def __call__(self, *args):
        return self.criterion(*args)
