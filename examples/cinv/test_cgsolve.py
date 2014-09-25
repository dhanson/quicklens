#!/usr/bin/env python
#
# quicklens/examples/cinv/test_cg_solve.py
# --
# simple test function for the conjugate gradient routines to solve Ax=b in quicklens/examples/cinv/cg_solve.py using several different preconditioners.

import numpy as np
import quicklens as ql

dim = 100 # matrix dimension.
np.random.seed(0)

class test_fwd():
    def __init__(self, mat):
        self.mat = mat
        
    def do(self, var):
        return np.dot(self.mat, var)

class test_pre_idt(test_fwd):
    """ identity matrix preconditioner. """
    def __init__(self, mat):
        self.mat = np.identity(np.shape(mat)[0])

class test_pre_dag(test_fwd):
    """ diagonal matrix preconditioner. """
    def __init__(self, mat):
        self.mat = np.diag( np.diag(mat) )
        
class test_pre_rnd(test_fwd):
    """ a random symmetric positive definite matrix preconditioner. """
    def __init__(self, mat):
        tmat = np.array(np.random.standard_normal(np.shape(mat)))
        self.mat = np.dot(np.transpose(tmat), tmat)

class test_pre_inv(test_fwd):
    """ a perfect preconditioner. """
    def __init__(self, mat):
        self.mat = np.linalg.pinv(mat) 

# generate a random symmetric positive definite matrix for A
mat = np.array(np.random.standard_normal([dim,dim]))
mat = np.dot(np.transpose(mat), mat) 

fwd_op = test_fwd(mat).do
pre_op_inv = test_pre_inv(mat).do
pre_op_idt = test_pre_idt(mat).do
pre_op_dag = test_pre_dag(mat).do
pre_op_rnd = test_pre_rnd(mat).do

# generate a random vector for b.
b = np.array(np.random.standard_normal(dim))

for (pre_op, label) in [ (pre_op_inv, 'perfect'),
                         (pre_op_idt, 'identity'),
                         (pre_op_dag, 'diagonal'),
                         (pre_op_rnd, 'random') ]:

    x = np.zeros(dim)
    (iter, eps) = ql.cinv.cg_solve.cg_solve_simple( x, b, fwd_op, pre_op, np.dot, iter_max=10000, eps_min=1.e-5 )

    sol = np.dot( np.linalg.pinv(mat), b )
    err = np.abs( x - sol )
    print 'preconditioner = ' + label
    print '    cgsolve completed. iteration =', iter, ' eps =', eps
    print '    max,min  x           = (', np.max(np.abs(sol)), ', ', np.min(np.abs(sol)), ')'
    print '    max,min abs(delta)   = (', np.max(err), ', ', np.min(err), ')'
    print '    "" abs(delta)/abs(x) = (', np.max(err / np.abs(sol)), ', ', np.min(err / np.abs(sol)), ')'
    print '    average abs(delta)   = ',  np.average(err)
    print ''
