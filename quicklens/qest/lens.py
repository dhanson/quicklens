# quicklens/qest/lens.py
# --
# this module contains quadratic estimators 'plm' for the CMB lensing
# potential, using weight functions for combinations of temperature (T) and
# polarization (E,B) from Hu et. al. http://arxiv.org/abs/astro-ph/0111606

import qest

class qest_plm_TT(qest.qest):
    """ temperature-temperature (TT) lensing potential estimator. """
    def __init__(self, cltt):
        """ initialize the TT lensing potential estimator.
             * cltt = lensed TT power spectrum.
        """
        self.cltt = cltt
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

    def wo_d2(self, l, lx, ly):
        return -0.5
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * l

class qest_plm_TT_s0(qest.qest):
    """ version of the temperature-temperature (TT) lensing potential estimator with spin-0 weights.
    equivalent to qest_plm_TT, but with different implementation of the weights.
    this class is meant to represent more closely the usual temperatre x (gradient temperature)
    description of the TT lensing estimator. """
    def __init__(self, cltt):
        """ initialize the spin-0 TT lensing estimator.
             * cltt = lensed TT power spectrum.
        """
        self.cltt = cltt
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

class qest_plm_TE(qest.qest):
    """ TE lensing potential estimator. """
    def __init__(self, clte):
        """ initialize the TE lensing potential estimator.
             * clte = lensed TE power spectrum.
        """
        self.clte = clte
        self.ntrm = 6

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_d4; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_d4; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_d4; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_d4; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        # dt e
        self.wl[4][0] = self.wo_d2; self.sl[4][0] = +0
        self.wl[4][1] = self.wc_ml; self.sl[4][1] = +1
        self.wl[4][2] = self.wo_ml; self.sl[4][2] = +1

        self.wl[5][0] = self.wo_d2; self.sl[5][0] = +0
        self.wl[5][1] = self.wc_ml; self.sl[5][1] = -1
        self.wl[5][2] = self.wo_ml; self.sl[5][2] = -1

    def wo_d2(self, l, lx, ly):
        return -0.50
    def wo_d4(self, l, lx, ly):
        return -0.25
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 ) * l

class qest_plm_TB(qest.qest):
    """ TB lensing potential estimator. """
    def __init__(self, clte):
        """ initialize the TB lensing potential estimator.
             * clte = lensed TE power spectrum.
        """
        self.clte = clte
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_di; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_mi; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_mi; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_di; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

    def wo_di(self, l, lx, ly):
        return +0.25j
    def wo_mi(self, l, lx, ly):
        return -0.25j
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clte)), self.clte, right=0 ) * l

class qest_plm_EE(qest.qest):
    """ EE lensing potential estimator. """
    def __init__(self, clee):
        """ initialize the EE lensing potential estimator.
             * clee = lensed EE power spectrum.
        """
        self.clee = clee
        self.ntrm = 8

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wo_d4; self.sl[0][0] = -2
        self.wl[0][1] = self.wc_ml; self.sl[0][1] = +3
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wo_d4; self.sl[1][0] = +2
        self.wl[1][1] = self.wc_ml; self.sl[1][1] = -3
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = +3
        self.wl[2][1] = self.wo_d4; self.sl[2][1] = -2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = -3
        self.wl[3][1] = self.wo_d4; self.sl[3][1] = +2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.wl[4][0] = self.wo_d4; self.sl[4][0] = +2
        self.wl[4][1] = self.wc_ml; self.sl[4][1] = -1
        self.wl[4][2] = self.wo_ml; self.sl[4][2] = +1

        self.wl[5][0] = self.wo_d4; self.sl[5][0] = -2
        self.wl[5][1] = self.wc_ml; self.sl[5][1] = +1
        self.wl[5][2] = self.wo_ml; self.sl[5][2] = -1

        self.wl[6][0] = self.wc_ml; self.sl[6][0] = -1
        self.wl[6][1] = self.wo_d4; self.sl[6][1] = +2
        self.wl[6][2] = self.wo_ml; self.sl[6][2] = +1

        self.wl[7][0] = self.wc_ml; self.sl[7][0] = +1
        self.wl[7][1] = self.wo_d4; self.sl[7][1] = -2
        self.wl[7][2] = self.wo_ml; self.sl[7][2] = -1

    def wo_d4(self, l, lx, ly):
        return -0.25
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * l

class qest_plm_EB(qest.qest):
    """ EB lensing potential estimator. """
    def __init__(self, clte):
        """ initialize the EB lensing potential estimator.
             * clte = lensed TE power spectrum.
        """
        self.clee = clee
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +3
        self.wl[0][1] = self.wo_di; self.sl[0][1] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -3
        self.wl[1][1] = self.wo_mi; self.sl[1][1] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wc_ml; self.sl[2][0] = -1
        self.wl[2][1] = self.wo_mi; self.sl[2][1] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wc_ml; self.sl[3][0] = +1
        self.wl[3][1] = self.wo_di; self.sl[3][1] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        self.npad_conv = 2 

    def wo_di(self, l, lx, ly):
        return +0.25j
    def wo_mi(self, l, lx, ly):
        return -0.25j
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.clee)), self.clee, right=0 ) * l

class qest_plm_ET(qest_plm_TE):
    """ ET lensing potential estimator. equivalent to qest_plm_TE, but with E in first index position. """
    def __init__(self, clte):
        """ initialize the ET lensing potential estimator.
             * clte = lensed TE power spectrum.
        """
        self.clte = clte
        self.ntrm = 6

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][1] = self.wc_ml; self.sl[0][1] = +3
        self.wl[0][0] = self.wo_d4; self.sl[0][0] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][1] = self.wc_ml; self.sl[1][1] = -3
        self.wl[1][0] = self.wo_d4; self.sl[1][0] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][1] = self.wc_ml; self.sl[2][1] = -1
        self.wl[2][0] = self.wo_d4; self.sl[2][0] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][1] = self.wc_ml; self.sl[3][1] = +1
        self.wl[3][0] = self.wo_d4; self.sl[3][0] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

        # dt e
        self.wl[4][1] = self.wo_d2; self.sl[4][1] = +0
        self.wl[4][0] = self.wc_ml; self.sl[4][0] = +1
        self.wl[4][2] = self.wo_ml; self.sl[4][2] = +1

        self.wl[5][1] = self.wo_d2; self.sl[5][1] = +0
        self.wl[5][0] = self.wc_ml; self.sl[5][0] = -1
        self.wl[5][2] = self.wo_ml; self.sl[5][2] = -1
    
class qest_plm_BT(qest_plm_TB):
    """ BT lensing potential estimator. equivalent to qest_plm_TB, but with B in the first index position. """
    def __init__(self, clte):
        """ initialize the BT lensing potential estimator.
             * clte = lensed TE power spectrum.
        """
        self.clte = clte
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][1] = self.wc_ml; self.sl[0][1] = +3
        self.wl[0][0] = self.wo_di; self.sl[0][0] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][1] = self.wc_ml; self.sl[1][1] = -3
        self.wl[1][0] = self.wo_mi; self.sl[1][0] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][1] = self.wc_ml; self.sl[2][1] = -1
        self.wl[2][0] = self.wo_mi; self.sl[2][0] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][1] = self.wc_ml; self.sl[3][1] = +1
        self.wl[3][0] = self.wo_di; self.sl[3][0] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1
    
class qest_plm_BE(qest_plm_EB):
    """ BE lensing potential estimator. equivalent to qest_plm_EB, but with B in first index position. """
    def __init__(self, clee):
        """ initialize the BE lensing potential estimator.
             * clee = lensed EE power spectrum.
        """
        self.clee = clee
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        # t de
        self.wl[0][1] = self.wc_ml; self.sl[0][1] = +3
        self.wl[0][0] = self.wo_di; self.sl[0][0] = -2
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][1] = self.wc_ml; self.sl[1][1] = -3
        self.wl[1][0] = self.wo_mi; self.sl[1][0] = +2
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][1] = self.wc_ml; self.sl[2][1] = -1
        self.wl[2][0] = self.wo_mi; self.sl[2][0] = +2
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][1] = self.wc_ml; self.sl[3][1] = +1
        self.wl[3][0] = self.wo_di; self.sl[3][0] = -2
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

class qest_xlm_TT(qest.qest):
    """ temperature-temperature (TT) lensing curl-mode estimator. """
    def __init__(self, cltt):
        """ initialize the TT lensing curl-mode estimator.
             * cltt = lensed TT power spectrum.
        """
        self.cltt = cltt
        self.ntrm = 4

        self.wl = { i : {} for i in xrange(0, self.ntrm) }
        self.sl = { i : {} for i in xrange(0, self.ntrm) }

        self.wl[0][0] = self.wc_ml; self.sl[0][0] = +1
        self.wl[0][1] = self.wo_d2; self.sl[0][1] = +0
        self.wl[0][2] = self.wo_ml; self.sl[0][2] = +1

        self.wl[1][0] = self.wc_ml; self.sl[1][0] = -1
        self.wl[1][1] = self.wo_n2; self.sl[1][1] = +0
        self.wl[1][2] = self.wo_ml; self.sl[1][2] = -1

        self.wl[2][0] = self.wo_d2; self.sl[2][0] = +0
        self.wl[2][1] = self.wc_ml; self.sl[2][1] = +1
        self.wl[2][2] = self.wo_ml; self.sl[2][2] = +1

        self.wl[3][0] = self.wo_n2; self.sl[3][0] = +0
        self.wl[3][1] = self.wc_ml; self.sl[3][1] = -1
        self.wl[3][2] = self.wo_ml; self.sl[3][2] = -1

    def wo_d2(self, l, lx, ly):
        return -0.5j
    def wo_n2(self, l, lx, ly):
        return +0.5j
    def wo_ml(self, l, lx, ly):
        return l
    def wc_ml(self, l, lx, ly):
        return np.interp( l, np.arange(0, len(self.cltt)), self.cltt, right=0 ) * l
