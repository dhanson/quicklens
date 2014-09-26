# quicklens/cinv/cg_solve.py
# --
# this module contains routines multiplication by a matrix inverse using conjugate gradients.
# the code is based on shewchuk's famous tutorial
#    "introduction to the conjugate gradient method without the agonizing pain"
#     http://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf

import numpy as np

def cg_solve_simple( x, b, fwd_op, pre_op, dot_op, iter_max=1000, eps_min=1.0e-5, roundoff=50 ):
    """ simple conjugate gradient loop to solve Ax=b, demonstrating use of cg_iterator. for information on arguments, see cg_solve. """
    my_cg_iterator = cg_iterator( x, b, fwd_op, pre_op, dot_op, roundoff )

    # initialize.
    (delta, resid) = my_cg_iterator.next()
    d0 = delta

    # loop
    for iter in xrange(1,iter_max+1):
        (delta, resid) = my_cg_iterator.next()
        if (delta < eps_min**2 * d0): break

    return (iter, delta/d0)

def cg_solve(x, b, fwd_op, pre_op, dot_op, criterion, apply_prep_op=None, apply_fini_op=None, roundoff=50):
    """ customizable conjugate gradient solver for Ax=b.
         * x             = storage buffer for solution (also used as initial guess, so typically initialize to zero).
         * b             = vector representing the right-hand-side of the equation above.
         * fwd_op        = function fwd_op(x) which returns Ax (the 'forward operation').
         * pre_op        = function pre_op(x) which approximates A^{-1}x (the 'preconditioner operation').
         * dot_op        = function dot_op(x,y) which returns dot product between x and y.
         * criterion     = function criterion(iteration number, current x estimate, residual, delta) which
                           returns true when the solution is deemed to be sufficiently converged.
         * apply_prep_op = operation to apply to b before beginning.
         * apply_fini_op = operation to apply to x after converging.
         * roundoff      = number of iterations between direct recomputations of the residual (to avoid roundoff error).

         note: fwd_op, pre_op and dot_op must not modify their arguments!
    """
    if (apply_prep_op is not None): apply_prep_op(b)
    
    cg_iter        = cg_iterator( x, b, fwd_op, pre_op, dot_op )
    (delta, resid) = cg_iter.next()

    iter = 0
    while criterion(iter, x, resid, delta) == False:
        (delta, resid) = cg_iter.next()
        iter += 1

    if (apply_fini_op is not None): apply_fini_op(x)

def cg_iterator( x, b, fwd_op, pre_op, dot_op, roundoff=50 ):
    """ conjugate gradient iterator for solving Ax=b.
    for information on arguments, see cg_solve. """
    residual  = b - fwd_op(x)
    searchdir = pre_op(residual)

    delta     = dot_op(residual, searchdir)
    
    iter = 0
    while True:
        assert( delta >= 0.0 ) #sanity check
        yield (delta, residual)
        
        searchfwd = fwd_op(searchdir)
        alpha     = delta / dot_op(searchdir, searchfwd)

        x += (searchdir * alpha)

        iter += 1
        if ( np.mod(iter, roundoff) == 0 ):
            residual = b - fwd_op(x)
        else:
            residual -= searchfwd * alpha

        tsearchdir = pre_op(residual)
        tdelta     = dot_op(residual, tsearchdir)

        searchdir *= (tdelta / delta)
        searchdir += tsearchdir

        delta = tdelta
