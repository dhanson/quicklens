# quicklens/mpi.py
# --
# this module contains hooks for parallel computing with the standard Message Passing Interface (MPI).
# currently it uses 'pypar' (https://github.com/daleroberts/pypar), which is itself a python wrapper
# around a C implementation of MPI. the only addition here is that if pypar cannot be loaded the
# wrapper functions are still defined, and behave as though the script were launched in serial mode
# with only one node.
#
# variables and functions
#     * rank       = index of this node.
#     * size       = total number of nodes.
#     * barrier()  = halt execution until all nodes have called barrier().
#     * finalize() = terminate the MPI session (otherwise if one node finishes before the others they may be killed as well).

import sys

try:
    import pypar
    rank = pypar.rank()
    size = pypar.size()
    barrier = pypar.barrier
    finalize = pypar.finalize

except ImportError, exc:
    sys.stderr.write("IMPORT ERROR: " + __file__ + " (" + str(exc) + "). Could not load pbs or pypar. MPI will not be used.\n")

    rank = 0
    size = 1

    def barrier():
        pass

    def finalize():
        pass
