# quicklens/qest/lens.py
# --
# this module contains quadratic estimators for the phi and psi potentials
# which give the CMB lensing deflection
#    d(n) = \grad \phi(n) + \curl \psi(n).
# they use weight functions W for combinations of temperature (T) and
# polarization (E,B) from Hu et. al. http://arxiv.org/abs/astro-ph/0111606

import numpy as np
import qest

class phi_TT(qest.qest):
    """ temperature-temperature (TT) lensing potential gradient-mode estimator. """
    def __init__(self, cltt):
        """ initialize the TT lensing potential gradient-mode estimator.
             * cltt = lensed TT power spectrum.
        """
        self.cltt = cltt
        self.lmax = len(cltt)-1
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +1
        self.wl[0][1] = self.wo_d2; self.sl[0][1] = +0
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -1
        self.wl[1][1] = self.wo_d2; self.sl[1][1] = +0
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wo_d2; self.sl[2][0] = +0
        self.wl[2][1] = self.wc_ml; self.sl[2][1] = +1
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wo_d2; self.sl[3][0] = +0
        self.wl[3][1] = self.wc_ml; self.sl[3][1] = -1
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

    def wo_d2(self, l, lx=None, ly=None):
        return -0.5
    def wo_ml(self, l, lx=None, ly=None):
        return np.sqrt(l*(l+1.))
    def wc_ml(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * np.sqrt(l*(l+1.))
    
class phi_TT_s0(qest.qest):
    """ version of the temperature-temperature (TT) lensing potential gradient-mode estimator with spin-0 weights.
    equivalent to phi_TT, but with different implementation of the weights.
    this class is meant to represent more closely the usual temperatre x (gradient temperature)
    description of the TT lensing estimator. """
    def __init__(self, cltt):
        """ initialize the spin-0 TT lensing estimator.
             * cltt = lensed TT power spectrum.
        """
        self.cltt = cltt
        self.lmax = len(cltt)-1
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : [0,0,0] for i in xrange(0,self.ntrm) }

        self.wl[0][0] = self.wc_lx
        self.wl[0][1] = self.wo_m1
        self.wl[0][2] = self.wo_lx

        self.wl[1][0] = self.wc_ly
        self.wl[1][1] = self.wo_m1
        self.wl[1][2] = self.wo_ly

        self.wl[2][0] = self.wo_m1
        self.wl[2][1] = self.wc_lx
        self.wl[2][2] = self.wo_lx

        self.wl[3][0] = self.wo_m1
        self.wl[3][1] = self.wc_ly
        self.wl[3][2] = self.wo_ly

    def wo_m1(self, l, lx, ly):
        return 1.0
    def wo_lx(self, l, lx, ly):
        return lx
    def wc_lx(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * lx
    def wo_ly(self, l, lx, ly):
        return ly
    def wc_ly(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * ly

class phi_TE(qest.qest):
    """ TE lensing potential gradient-mode estimator. """
    def __init__(self, clte):
        """ initialize the TE lensing potential gradient-mode estimator.
             * clte = lensed TE power spectrum.
        """
        self.clte = clte
        self.lmax = len(clte)-1
        self.ntrm = 6

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_m3; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_d4; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_m1; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_m3; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_d4; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_m1; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_m2; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_d4; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_m1; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_m2; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_d4; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_m1; self.sl[3][2] = -1

        # dt e
        self.wl[4][0] = self.wo_d2; self.sl[4][0] = +0
        self.wl[4][1] = self.wc_m1; self.sl[4][1] = +1
        self.wl[4][2] = self.wo_m1; self.sl[4][2] = +1

        self.wl[5][0] = self.wo_d2; self.sl[5][0] = +0
        self.wl[5][1] = self.wc_m1; self.sl[5][1] = -1
        self.wl[5][2] = self.wo_m1; self.sl[5][2] = -1

    def wo_d2(self, l, lx=None, ly=None):
        return -0.50
    def wo_d4(self, l, lx=None, ly=None):
        return -0.25
    def wo_m1(self, l, lx=None, ly=None):
        return np.sqrt(l*(l+1.))
    def wc_m1(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 ) * np.sqrt( l*(l+1.) )
    def wc_m2(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 ) * np.nan_to_num( np.sqrt( (l+2.)*(l-1.) ) )
    def wc_m3(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 ) * np.nan_to_num( np.sqrt( (l+3.)*(l-2.) ) )

class phi_TB(qest.qest):
    """ TB lensing potential gradient-mode estimator. """
    def __init__(self, clte):
        """ initialize the TB lensing potential gradient-mode estimator.
             * clte = lensed TE power spectrum.
        """
        self.clte = clte
        self.lmax = len(clte)-1
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_m3; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_di; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_m1; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_m3; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_mi; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_m1; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_m2; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_mi; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_m1; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_m2; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_di; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_m1; self.sl[3][2] = -1

    def wo_di(self, l, lx=None, ly=None):
        return +0.25j
    def wo_mi(self, l, lx=None, ly=None):
        return -0.25j
    def wo_m1(self, l, lx=None, ly=None):
        return np.sqrt(l*(l+1.))
    def wc_m2(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 ) * np.nan_to_num( np.sqrt( (l+2.)*(l-1.) ) )
    def wc_m3(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 ) * np.nan_to_num( np.sqrt( (l+3.)*(l-2.) ) )

class phi_EE(qest.qest):
    """ EE lensing potential gradient-mode estimator. """
    def __init__(self, clee):
        """ initialize the EE lensing potential gradient-mode estimator.
             * clee = lensed EE power spectrum.
        """
        self.clee = clee
        self.lmax = len(clee)-1
        self.ntrm = 8

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # first term.
        self.wl[0][0] = self.wo_d4; self.sl[0][0] = -2
        self.wl[0][1] = self.wc_m3; self.sl[0][1] = +3
        self.wl[0][2] = self.wo_m1; self.sl[0][2] = +1

        self.wl[1][0] = self.wo_d4; self.sl[1][0] = +2
        self.wl[1][1] = self.wc_m3; self.sl[1][1] = -3
        self.wl[1][2] = self.wo_m1; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_m3; self.sl[2][0] = +3
        self.wl[2][1] = self.wo_d4; self.sl[2][1] = -2
        self.wl[2][2] = self.wo_m1; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_m3; self.sl[3][0] = -3
        self.wl[3][1] = self.wo_d4; self.sl[3][1] = +2
        self.wl[3][2] = self.wo_m1; self.sl[3][2] = -1

        # second term.
        self.wl[4][0] = self.wo_d4; self.sl[4][0] = +2
        self.wl[4][1] = self.wc_m2; self.sl[4][1] = -1
        self.wl[4][2] = self.wo_m1; self.sl[4][2] = +1

        self.wl[5][0] = self.wo_d4; self.sl[5][0] = -2
        self.wl[5][1] = self.wc_m2; self.sl[5][1] = +1
        self.wl[5][2] = self.wo_m1; self.sl[5][2] = -1

        self.wl[6][0] = self.wc_m2; self.sl[6][0] = -1
        self.wl[6][1] = self.wo_d4; self.sl[6][1] = +2
        self.wl[6][2] = self.wo_m1; self.sl[6][2] = +1

        self.wl[7][0] = self.wc_m2; self.sl[7][0] = +1
        self.wl[7][1] = self.wo_d4; self.sl[7][1] = -2
        self.wl[7][2] = self.wo_m1; self.sl[7][2] = -1

    def wo_d4(self, l, lx=None, ly=None):
        return -0.25
    def wo_m1(self, l, lx=None, ly=None):
        return np.sqrt(l*(l+1.))
    def wc_m2(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * np.nan_to_num( np.sqrt( (l+2.)*(l-1.) ) )
    def wc_m3(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * np.nan_to_num( np.sqrt( (l+3.)*(l-2.) ) )

class phi_EB(qest.qest):
    """ EB lensing potential gradient-mode estimator. """
    def __init__(self, clee):
        """ initialize the EB lensing potential gradient-mode estimator.
             * clte = lensed TE power spectrum.
        """
        self.clee = clee
        self.lmax = len(clee)-1
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_m3; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_di; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_m1; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_m3; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_mi; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_m1; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_m2; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_mi; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_m1; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_m2; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_di; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_m1; self.sl[3][2] = -1

    def wo_di(self, l, lx=None, ly=None):
        return +0.25j
    def wo_mi(self, l, lx=None, ly=None):
        return -0.25j
    def wo_m1(self, l, lx=None, ly=None):
        return np.sqrt( l*(l+1.) )
    def wc_m2(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * np.nan_to_num( np.sqrt( (l+2.)*(l-1.) ) )
    def wc_m3(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * np.nan_to_num( np.sqrt( (l+3.)*(l-2.) ) )

class phi_ET(phi_TE):
    """ ET lensing potential gradient-mode estimator. equivalent to phi_TE, but with E in first index position. """
    def __init__(self, clte):
        """ initialize the ET lensing potential gradient-mode estimator.
             * clte = lensed TE power spectrum.
        """
        super(phi_ET, self).__init__(clte)
        for i in xrange(0,self.ntrm):
            twl = self.wl[i][0]
            tsl = self.sl[i][0]
            
            self.wl[i][0] = self.wl[i][1]
            self.sl[i][0] = self.sl[i][1]

            self.wl[i][1] = twl
            self.sl[i][1] = tsl
        
class phi_BT(phi_TB):
    """ BT lensing potential gradient-mode estimator. equivalent to phi_TB, but with B in the first index position. """
    def __init__(self, clte):
        """ initialize the BT lensing potential gradient-mode estimator.
             * clte = lensed TE power spectrum.
        """
        super(phi_BT, self).__init__(clte)
        for i in xrange(0,self.ntrm):
            twl = self.wl[i][0]
            tsl = self.sl[i][0]
            
            self.wl[i][0] = self.wl[i][1]
            self.sl[i][0] = self.sl[i][1]

            self.wl[i][1] = twl
            self.sl[i][1] = tsl
    
class phi_BE(phi_EB):
    """ BE lensing potential gradient-mode estimator. equivalent to phi_EB, but with B in first index position. """
    def __init__(self, clee):
        """ initialize the BE lensing potential gradient-mode estimator.
             * clee = lensed EE power spectrum.
        """
        super(phi_BE, self).__init__(clee)
        for i in xrange(0,self.ntrm):
            twl = self.wl[i][0]
            tsl = self.sl[i][0]
            
            self.wl[i][0] = self.wl[i][1]
            self.sl[i][0] = self.sl[i][1]

            self.wl[i][1] = twl
            self.sl[i][1] = tsl

class psi_TT(qest.qest):
    """ temperature-temperature (TT) lensing potential curl-mode estimator. """
    def __init__(self, cltt):
        """ initialize the TT lensing potential curl-mode estimator.
             * cltt = lensed TT power spectrum.
        """
        self.cltt = cltt
        self.lmax = len(cltt)-1
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc_m1; self.sl[0][0] = +1
        self.wl[0][1] = self.wo_d2; self.sl[0][1] = +0
        self.wl[0][2] = self.wo_m1; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_m1; self.sl[1][0] = -1
        self.wl[1][1] = self.wo_n2; self.sl[1][1] = +0
        self.wl[1][2] = self.wo_m1; self.sl[1][2] = -1

        self.wl[2][0] = self.wo_d2; self.sl[2][0] = +0
        self.wl[2][1] = self.wc_m1; self.sl[2][1] = +1
        self.wl[2][2] = self.wo_m1; self.sl[2][2] = +1

        self.wl[3][0] = self.wo_n2; self.sl[3][0] = +0
        self.wl[3][1] = self.wc_m1; self.sl[3][1] = -1
        self.wl[3][2] = self.wo_m1; self.sl[3][2] = -1

    def wo_d2(self, l, lx=None, ly=None):
        return -0.5j
    def wo_n2(self, l, lx=None, ly=None):
        return +0.5j
    def wo_m1(self, l, lx=None, ly=None):
        return np.sqrt(l*(l+1.))
    def wc_m1(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * np.sqrt(l*(l+1.))

class blen_EP(qest.qest):
    """ E-mode-phi (EP) estimator for the lensed B-mode. """
    def __init__(self, clee, clpp):
        """ initialize the EP lensing B-mode estimator.
             * clee = lensed E-mode power spectrum.
             * clpp = gradient-mode lensing potential power spectrum.
        """
        self.clee = clee
        self.clpp = clpp

        assert(len(clee) == len(clpp))
        self.lmax = len(clee)-1
        
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_m3; self.sl[0][0] = +3
        self.wl[0][1] = self.wp_m1; self.sl[0][1] = -1
        self.wl[0][2] = self.wo_di; self.sl[0][2] = +2

        self.wl[1][0] = self.wc_m3; self.sl[1][0] = -3
        self.wl[1][1] = self.wp_m1; self.sl[1][1] = +1
        self.wl[1][2] = self.wo_mi; self.sl[1][2] = -2

        self.wl[2][0] = self.wc_m2; self.sl[2][0] = -1
        self.wl[2][1] = self.wp_m1; self.sl[2][1] = -1
        self.wl[2][2] = self.wo_mi; self.sl[2][2] = -2

        self.wl[3][0] = self.wc_m2; self.sl[3][0] = +1
        self.wl[3][1] = self.wp_m1; self.sl[3][1] = +1
        self.wl[3][2] = self.wo_di; self.sl[3][2] = +2

    def wo_di(self, l, lx=None, ly=None):
        return -0.25j
    def wo_mi(self, l, lx=None, ly=None):
        return +0.25j
    def wc_m2(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * np.nan_to_num( np.sqrt( (l+2.)*(l-1.) ) )
    def wc_m3(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * np.nan_to_num( np.sqrt( (l+3.)*(l-2.) ) )
    def wp_m1(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clpp)), self.clpp, right=0 ) * np.sqrt( l*(l+1.) )

class blen_EX(qest.qest):
    """ E-mode-psi (EX) estimator for the lensed B-mode. """
    def __init__(self, clee, clpp):
        """ initialize the EP lensing B-mode estimator.
             * clee = lensed E-mode power spectrum.
             * clpp = curl-mode lensing potential power spectrum.
        """
        self.clee = clee
        self.clpp = clpp

        assert(len(clee) == len(clpp))
        self.lmax = len(clee)-1
        
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_m3; self.sl[0][0] = +3
        self.wl[0][1] = self.wp_m1; self.sl[0][1] = -1
        self.wl[0][2] = self.wo_di; self.sl[0][2] = +2

        self.wl[1][0] = self.wc_m3; self.sl[1][0] = -3
        self.wl[1][1] = self.wp_m1; self.sl[1][1] = +1
        self.wl[1][2] = self.wo_di; self.sl[1][2] = -2

        self.wl[2][0] = self.wc_m2; self.sl[2][0] = -1
        self.wl[2][1] = self.wp_m1; self.sl[2][1] = -1
        self.wl[2][2] = self.wo_mi; self.sl[2][2] = -2

        self.wl[3][0] = self.wc_m2; self.sl[3][0] = +1
        self.wl[3][1] = self.wp_m1; self.sl[3][1] = +1
        self.wl[3][2] = self.wo_mi; self.sl[3][2] = +2

    def wo_di(self, l, lx=None, ly=None):
        return -0.25j
    def wo_mi(self, l, lx=None, ly=None):
        return +0.25j
    def wc_m2(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * np.nan_to_num( np.sqrt( (l+2.)*(l-1.) ) )
    def wc_m3(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * np.nan_to_num( np.sqrt( (l+3.)*(l-2.) ) )
    def wp_m1(self, l, lx=None, ly=None):
        return np.interp( l, np.arange(0, len(self.clpp)), self.clpp, right=0 ) * np.sqrt( l*(l+1.) )
