# quicklens/cinv/cd_solve.py
# --
# this module contains routines multiplication by a matrix inverse using conjugate directions,
# with options for truncation and partial restart

import numpy as np

def tpr(t, p, r):
    """ truncation and partial-restart template function. """
    return (lambda i : max(0, i - max(p, int(min(t, np.mod(i, r))))))

def tr_cg(i):
    """ truncation and restart function for conjugate gradients. """
    return i-1

def tr_cd(i):
    """ truncation and restart function for conjugate descent. """
    return 0

class cache_mem(dict):
    def __init__(self):
        pass

    def store(self, key, data):
        [dTAd_inv, searchdirs, searchfwds] = data
        self[key] = [dTAd_inv, searchdirs, searchfwds]

    def restore(self, key):
        return self[key]

    def remove(self, key):
        del self[key]

    def trim(self, keys):
        assert( set(keys).issubset(self.keys()) )
        for key in (set(self.keys()) - set(keys)):
            del self[key]

def cd_solve(x, b, fwd_op, pre_ops, dot_op, criterion, tr=tr_cg, cache=cache_mem(), roundoff=25):
    """ customizable conjugate directions solver for Ax=b.
         * x             = storage buffer for solution (also used as initial guess, so typically initialize to zero).
         * b             = vector representing the right-hand-side of the equation above.
         * fwd_op        = function fwd_op(x) which returns Ax (the 'forward operation').
         * pre_op        = function pre_op(x) which approximates A^{-1}x (the 'preconditioner operation').
         * dot_op        = function dot_op(x,y) which returns dot product between x and y.
         * criterion     = function criterion(iteration number, current x estimate, residual, delta) which
                           returns true when the solution is deemed to be sufficiently converged.
         * tr            = truncation / restart function. for each iteration 'i' of the descent, the search direction
                           is orthogonalized with respect to the directions for previous iterations indexed by tr(i), tr(i)+1, ..., i-1.
                           suggested form for tr(i) is truncated partial restart (TPR) with

                               tr(i) = i - max(P, min( T, mod(i, R) ))

                           where P=minimum truncation, T=maximum truncation length, R=restart period.
                           tr(i) must be monotonically increasing.
         * cache         = cacher object for previous search directions.
         * roundoff      = number of iterations between direct recomputations of the residual (to avoid roundoff error).

         note: fwd_op, pre_op and dot_op must not modify their arguments!
    """
    n_pre_ops = len(pre_ops)

    residual   = b - fwd_op(x)
    searchdirs = [op(residual) for op in pre_ops]
    
    it = 0
    while criterion(it, x, residual) == False:
        searchfwds = [fwd_op(searchdir) for searchdir in searchdirs]
        deltas     = [dot_op(searchdir, residual) for searchdir in searchdirs]

        #calculate (D^T A D)^{-1}
        dTAd = np.zeros( (n_pre_ops, n_pre_ops) )
        for ip1 in range(0, n_pre_ops):
            for ip2 in range(0, ip1+1):
                dTAd[ip1, ip2] = dTAd[ip2, ip1] = dot_op(searchdirs[ip1], searchfwds[ip2])
        dTAd_inv = np.linalg.inv(dTAd)

        # search.
        alphas = np.dot( dTAd_inv, deltas )
        for (searchdir, alpha) in zip( searchdirs, alphas ):
            x += searchdir * alpha

        # append to cache.
        cache.store( it, [dTAd_inv, searchdirs, searchfwds] )

        # update residual
        it += 1
        if ( np.mod(it, roundoff) == 0 ):
            residual = b - fwd_op(x)
        else:
            for (searchfwd, alpha) in zip( searchfwds, alphas ):
                residual -= searchfwd * alpha

        # initial choices for new search directions.
        searchdirs = [pre_op(residual) for pre_op in pre_ops]

        # orthogonalize w.r.t. previous searches.
        prev_iters = range( tr(it), it )
        
        for titer in prev_iters:
            [prev_dTAd_inv, prev_searchdirs, prev_searchfwds] = cache.restore(titer)

            for searchdir in searchdirs:
                proj  = [ dot_op(searchdir, prev_searchfwd) for prev_searchfwd in prev_searchfwds ]
                betas = np.dot( prev_dTAd_inv, proj )

                for (beta, prev_searchdir) in zip( betas, prev_searchdirs):
                    searchdir -= prev_searchdir * beta

        # clear old keys from cache
        cache.trim( range( tr(it+1), it ) )
        
    return it
