# quicklens/spec.py
# --
# this module contains classes and subroutines
# for working with the power spectra of temperature and polarization maps.
# overview of classes and functions:
#     * bl                = returns the transfer function for a symmetric Gaussian beam.
#     * nl                = returns the beam-deconvolved noise power spectrum for white map noise.
#     * get_camb_scalcl   = loads the scalar cl files produced by CAMB.
#     * get_camb_lensedcl = loads the lensed cl files produced by CAMB.
#     * camb_clfile       = base class to load and encapsulate cls produced by CAMB.
#     * lcl               = class to hold a 1D power spectrum with uniform-width bins with delta L = 1.
#                           (useful to compress a full 2D power spectrum with minimal loss of information.)
#     * bcl               = class to hold a binned 1D power spectrum.
#     * clvec             = class to hold a theory power spectrum (with delta L = 1).
#     * clmat_teb         = class to hold a 3x3 matrix of theory power spectra between T, E and B.
#     * rcfft2cl          = function to compute the 1D auto- or cross-spectra of real and complex 2D Fourier modes.
#     * tebfft2cl         = function to compute the 1D auto- or cross-spectra of 2D T, E, and B Fourier modes.
#     * cross_cl          = convenience wrapper around rcfft2cl and tebfft2cl.
#     * cl2cfft           = function to paste a 1D power spectrum onto the 2D Fourier plane.
#     * plot_cfft_cl2d    = function to make a 2D plot of power for a set of Fourier modes.

import os, copy, glob, hashlib
import numpy as np
import pylab as pl

import util
import maps

def bl(fwhm_arcmin, lmax):
    """ returns the map-level transfer function for a symmetric Gaussian beam.
         * fwhm_arcmin      = beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax             = maximum multipole.
    """
    ls = np.arange(0, lmax+1)
    return np.exp( -(fwhm_arcmin * np.pi/180./60.)**2 / (16.*np.log(2.)) * ls*(ls+1.) )

def nl(noise_uK_arcmin, fwhm_arcmin, lmax):
    """ returns the beam-deconvolved noise power spectrum in units of uK^2 for
         * noise_uK_arcmin = map noise level in uK.arcmin
         * fwhm_arcmin     = beam full-width-at-half-maximum (fwhm) in arcmin.
         * lmax            = maximum multipole.
    """
    return (noise_uK_arcmin * np.pi/180./60.)**2 / bl(fwhm_arcmin, lmax)**2

def get_camb_scalcl(fname=None, prefix=None, lmax=None):
    """ loads and returns a "scalar Cls" file produced by CAMB (camb.info).

         * (optional) fname  = file name to load.
         * (optional) prefix = directory in quicklens/data/cl directory to pull the *_scalCls.dat from. defaults to 'planck_wp_highL' (only used if fname==None).
         * (optional) lmax   = maximum multipole to load (all multipoles in file will be loaded by default).
    """
    if fname == None:
        basedir = os.path.dirname(__file__)
    
        if prefix == None:
            prefix = "planck_wp_highL"
        fname = basedir + "/data/cl/" + prefix + "/*_scalCls.dat"
    else:
        assert( prefix == None )
        
    tf = glob.glob( fname )
    assert(len(tf) == 1),"No filename matching {0} found!".format(fname)
    
    return camb_clfile( tf[0], lmax=lmax )

def get_camb_lensedcl(fname=None, prefix=None, lmax=None):
    """ loads and returns a "lensed Cls" file produced by CAMB (camb.info).

         * (optional) fname  = file name to load.
         * (optional) prefix = directory in quicklens/data/cl directory to pull the *_lensedCls.dat from. defaults to 'planck_wp_highL' (only used if fname==None).
         * (optional) lmax   = maximum multipole to load (all multipoles in file will be loaded by default).
    """
    if fname ==None:
        basedir = os.path.dirname(__file__)

        if prefix == None:
            prefix = "planck_wp_highL"
        fname = basedir + "/data/cl/" + prefix + "/*_lensedCls.dat"

    tf = glob.glob( fname )
    assert(len(tf) == 1)
    return camb_clfile( tf[0], lmax=lmax )

def is_camb_clfile(obj):
    """ check (by ducktyping) if obj is a Cls file produced by CAMB """
    if not hasattr(obj, 'lmax'):
        return False
    if not hasattr(obj, 'ls'):
        return False
    return set(object.__dict__.keys()).issubset( set( ['lmax', 'ls', 'cltt', 'clee', 'clte', 'clpp', 'cltp', 'clbb', 'clep', 'cleb' ] ) )

class camb_clfile(object):
    """ class to load and store Cls from the output files produced by CAMB. """
    def __init__(self, tfname, lmax=None):
        """ load Cls.
             * tfname           = file name to load from.
             * (optional) lmax  = maximum multipole to load (all multipoles in file will be loaded by default).
        """
        tarray = np.loadtxt(tfname)
        lmin   = tarray[0, 0]
        assert(int(lmin)==lmin)
        lmin = int(lmin)

        if lmax == None:
            lmax = np.shape(tarray)[0]-lmin+1
            if lmax > 10000:
                lmax = 10000
            else:
                assert(tarray[-1, 0] == lmax)
        assert( (np.shape(tarray)[0]+1) >= lmax )

        ncol = np.shape(tarray)[1]
        ell  = np.arange(lmin, lmax+1, dtype=np.float)

        self.lmax = lmax
        self.ls   = np.concatenate( [ np.arange(0, lmin), ell ] )
        if ncol == 5:                                                                            # _lensedCls
            self.cltt = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),1]*2.*np.pi/ell/(ell+1.) ] )
            self.clee = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),2]*2.*np.pi/ell/(ell+1.) ] )
            self.clbb = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),3]*2.*np.pi/ell/(ell+1.) ] )
            self.clte = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),4]*2.*np.pi/ell/(ell+1.) ] )

        elif ncol == 6:                                                                          # _scalCls
            tcmb  = 2.726*1e6 #uK

            self.cltt = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),1]*2.*np.pi/ell/(ell+1.) ] )
            self.clee = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),2]*2.*np.pi/ell/(ell+1.) ] )
            self.clte = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),3]*2.*np.pi/ell/(ell+1.) ] )
            self.clpp = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),4]/ell**4/tcmb**2 ] )
            self.cltp = np.concatenate( [ np.zeros(lmin), tarray[0:(lmax-lmin+1),5]/ell**3/tcmb ] )

    def copy(self, lmax=None, lmin=None):
        """ clone this object.
             * (optional) lmax = restrict copy to L<=lmax.
             * (optional) lmin = set spectra in copy to zero for L<lmin.
        """
        if (lmax == None):
            return copy.deepcopy(self)
        else:
            assert( lmax <= self.lmax )
            ret      = copy.deepcopy(self)
            ret.lmax = lmax
            ret.ls   = np.arange(0, lmax+1)
            for k, v in self.__dict__.items():
                if k[0:2] == 'cl':
                    setattr( ret, k, copy.deepcopy(v[0:lmax+1]) )

            if lmin != None:
                assert( lmin <= lmax )
                for k in self.__dict__.keys():
                    if k[0:2] == 'cl':
                        getattr( ret, k )[0:lmin] = 0.0
            return ret

    def hashdict(self):
        """ return a dictionary uniquely associated with the contents of this clfile. """
        ret = {}
        for attr in ['lmax', 'cltt', 'clee', 'clte', 'clpp', 'cltp', 'clbb', 'clep', 'cleb' ]:
            if hasattr(self, attr):
                ret[attr] = getattr(self, attr)
        return ret

    def __add__(self, other):
        """ sum two clfile objects.
             * lmax must be the same for both.
             * spectra which are not common between the two are ignored.
        """
        if is_camb_clfile(other):
            assert( self.lmax == other.lmax )
            ret = self.copy()
            zs  = np.zeros(self.lmax+1)
            for attr in ['cltt', 'clee', 'clte', 'clpp', 'cltp', 'clbb', 'clep', 'cleb' ]:
                if (hasattr(self, attr) or hasattr(other, attr)):
                    setattr(ret, attr, getattr(self, attr, zs) + getattr(other, attr, zs) )
            return ret
        else:
            assert(0)

    def __eq__(self, other):
        """ compare two clfile objects. """
        try:
            for key in self.__dict__.keys()+other.__dict__.keys():
                if type(self.__dict__[key]) == np.ndarray:
                    assert( np.all( self.__dict__[key] == other.__dict__[key] ) )
                else:
                    assert( self.__dict__[key] == other.__dict__[key] )                
        except:
            return False

        return True

    def plot(self, spec='cltt', p=pl.plot, t=lambda l:1., **kwargs):
        """ plot the spectrum
             * spec = spectrum to display (e.g. cltt, clee, clte, etc.)
             * p    = plotting function to use p(x,y,**kwargs)
             * t    = scaling to apply to the plotted Cl -> t(l)*Cl
        """
        p( self.ls, t(self.ls) * getattr(self, spec), **kwargs )

def is_lcl(obj):
    """ check (by ducktyping) whether obj is an lcl object. """
    return ( hasattr(obj, 'lmax') and hasattr(obj, 'nm') and hasattr(obj, 'cl') )

class lcl(object):
    """ class to hold a 1-dimensional power spectrum, binned from a 2D FFT with constant delta L=1 bins. contains attributes:
         * lmax = maximum multipole l.
         * cl   = the binned power spectrum cl.
         * nm   = number of modes in each bin (number or pixels in each annulus).
    """
    def __init__(self, lmax, fc):
        """ lcl constructor.
             * lmax          = maximum multipole l.
             * fc            = complex FFT (cfft) object to calculate the power spectrum from.
        """
        self.lmax = lmax

        ell = fc.get_ell().flatten()
        self.nm, bins = np.histogram(ell, bins=np.arange(0, lmax+1, 1))
        self.cl, bins = np.histogram(ell, bins=np.arange(0, lmax+1, 1), weights=fc.fft.flatten())
        self.cl[np.nonzero(self.nm)] /= self.nm[np.nonzero(self.nm)]

    def is_compatible(self, other):
        """ check if this object can be added, subtracted, etc. with other. """
        return ( is_lcl(other) and (self.lmax == other.lmax) and np.all(self.nm == other.nm) )

    def __add__(self, other):
        assert( self.is_compatible(other) )
        ret = copy.deepcopy(self)
        ret.cl += other.cl
        return ret
    def __sub__(self, other):
        assert( self.is_compatible(other) )
        ret = copy.deepcopy(self)
        ret.cl -= other.cl
        return ret
    def __mul__(self, other):
        if np.isscalar(other):
            ret = copy.deepcopy(self)
            ret.cl *= other
            return ret
        elif is_lcl(other):
            assert( self.is_compatible(other) )
            ret = copy.deepcopy(self)
            ret.cl[np.nonzero(self.nm)] *= other.cl[np.nonzero(self.nm)]
            return ret
        else:
            assert(0)
    def __div__(self, other):
        if np.isscalar(other):
            ret = copy.deepcopy(self)
            ret.cl /= other
            return ret
        elif is_lcl(other):
            assert( self.is_compatible(other) )
            ret = copy.deepcopy(self)
            ret.cl[np.nonzero(self.nm)] /= other.cl[np.nonzero(self.nm)]
            return ret
        else:
            assert(0)
    def get_ml(self, lbins, t=lambda l : 1.):
        """ rebins this spectrum with non-uniform binning as a bcl object.
             * lbins        = list definining the bin edges [lbins[0], lbins[1]], [lbins[1], lbins[2]], ...
             * (optional) w = l-dependent scaling to apply when accumulating into bins (in addition to number of modes in each bin).
        """
        l = np.arange(0, self.lmax+1, 1)
        l = 0.5*(l[:-1] + l[1:]) # get bin centers
        t = t(l)
        
        modes = np.nan_to_num(self.nm)
        
        norm, bins = np.histogram(l, bins=lbins, weights=modes) # get number of modes in each l-bin.
        spec, bins = np.histogram(l, bins=lbins, weights=t*modes*np.nan_to_num(self.cl)) # bin the spectrum.

        # normalize the spectrum
        spec[np.nonzero(norm)] /= norm[np.nonzero(norm)]
        
        return bcl(lbins, {'cl' : spec})

class bcl(object):
    """ binned power spectrum. contains attributes:
         * specs = dictionary, contaning binned spectra.
         * lbins = list defining the bin edges [lbins[0], lbins[1]], [lbins[1], lbins[2]], ...
         * ls    = bin centers, given by average of left and right edges.
    """
    def __init__(self, lbins, specs):
        self.lbins = lbins
        self.specs = specs

        self.ls    = 0.5*(lbins[0:-1] + lbins[1:]) # get bin centers

    def __getattr__(self, spec):
        try:
            return self.specs[spec]
        except KeyError:
            raise AttributeError(spec)

    def __mul__(self, fac):
        ret = copy.deepcopy(self)

        for spec in ret.specs.keys():
            ret.specs[spec][:] *= fac

        return ret

    def __div__(self, other):
        ret = copy.deepcopy(self)

        if np.isscalar(other):
            for spec in ret.specs.keys():
                ret.specs[spec][:] /= other
        elif (hasattr(other, 'lbins') and hasattr(other, 'specs')):
            assert( np.all(self.lbins == other.lbins) )
            for spec in ret.specs.keys():
                ret.specs[spec][:] /= other.specs[spec][:]

        return ret

    def __add__(self, other):
        if (hasattr(other, 'lbins') and hasattr(other, 'specs')):
            assert( np.all(self.lbins == other.lbins) )

            ret = copy.deepcopy(self)
            for spec in ret.specs.keys():
                ret.specs[spec][:] += other.specs[spec][:]
            return ret
        else:
            assert(0)

    def __sub__(self, other):
        if (hasattr(other, 'lbins') and hasattr(other, 'specs')):
            assert( np.all(self.lbins == other.lbins) )

            ret = copy.deepcopy(self)
            for spec in ret.specs.keys():
                ret.specs[spec][:] -= other.specs[spec][:]
            return ret
        else:
            assert(0)

    def __iadd__(self, other):
        assert( np.all(self.lbins == other.lbins) )
        assert( self.specs.keys() == other.specs.keys() )

        for spec in self.specs.keys():
            self.specs[spec][:] += other.specs[spec]

        return self

    def get_ml(self, lbins, t=lambda l : 1.):
        """ rebin this spectrum to wider lbins.
        currently only implemented for trivial case where lbins are the same as those used by this object. """
        if np.all(self.lbins == lbins):
            return self
        else:
            assert(0)

    def plot(self, spec='cl', p=pl.plot, t=lambda l:1., **kwargs):
        """ plot the binned spectrum
             * (optional) spec = spectrum to display (e.g. cltt, clee, clte, etc.). defaults to 'cl'.
             * (optional) p    = plotting function to use p(x,y,**kwargs).
             * (optional) t    = scaling to apply to the plotted Cl -> t(l)*Cl.
        """
        p( self.ls, t(self.ls) * self.specs[spec], **kwargs )

def is_cl(obj):
    """ check (by ducktyping) if obj is a cl. """
    if not ( hasattr(obj, 'lmax') and hasattr(obj, 'clmat') ):
        return False
    return obj.clmat.shape == (obj.lmax+1)

class clvec(object):
    """ class to encapsulate a power spectrum vector c[l] for a single field. contains operator overloading for multiplying by a fourier transform object. """
    def __init__(self, clvec):
        self.lmax  = len(clvec)-1
        self.clvec = clvec.copy()

    def clone(self, lmax=None):
        """ clone this cl object, optionally restricting to 0<=L<=lmax. """
        if lmax == None:
            lmax = self.lmax
        assert(lmax <= self.lmax)
        ret = sinv_filt( np.zeros(lmax+1) )
        ret.clvec[:] = self.clvec[0:lmax+1]

        return ret

    def __add__(self, other):
        if ( hasattr(other, 'fft') and hasattr(other, 'get_ell') ):
            ret = other.copy()
            ell = other.get_ell()

            ret.fft[:,:]  += np.interp( ell.flatten(), np.arange(0, self.lmax+1), self.clvec[:], right=0 ).reshape(ell.shape)
            return ret
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            assert( (self.lmax+1) <= len(other) )
            return clvec( self.clvec + other[0:self.lmax+1] )
        else:
            assert(0)

    def __mul__(self, other):
        if False:
            pass
        elif np.isscalar(other):
            return clvec( self.clvec * other )
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            assert( (self.lmax+1) <= len(other) )
            return clvec( self.clvec * other[0:lmax+1] )

        elif ( hasattr(other, 'fft') and hasattr(other, 'get_ell') ):
            ret = other.copy()
            ell = other.get_ell()
        
            def fftxcl(fft, cl):
                return fft * np.interp( ell.flatten(), np.arange(0, len(cl)), cl, right=0 ).reshape(fft.shape)
        
            ret.fft[:,:]  = fftxcl( other.fft, self.clvec[:] )
            return ret
        else:
            assert(0)
        
    def inverse(self):
        """ return a new clvec object, containing the 1/clvec (containing zero where clvec==0). """
        ret = clvec( np.zeros(self.lmax+1) )
        ret.clvec[ np.nonzero(self.clvec) ] = 1./self.clvec[ np.nonzero(self.clvec) ]
        return ret

    def cholesky(self):
        """ return a new clvec object, containing sqrt(clvec). """
        return clvec( np.sqrt(self.clvec) )

def is_clmat_teb(obj):
    """ check (by ducktyping) if obj is a clmat_teb object. """
    if not ( hasattr(obj, 'lmax') and hasattr(obj, 'clmat') ):
        return False
    return obj.clmat.shape == (obj.lmax+1, 3, 3)

class clmat_teb(object):
    """ class to encapsulate the 3x3 covariance matrix at each multipole for a set of T, E and B auto- and cross-spectra. contains operator overloading for multiplying by a tebfft object. """
    def __init__(self, cl):
        """ initializes this clmat_teb object using the power spectra cl.cltt, cl.clte, etc.
        spectra which are not present in cl are assumed to be zero. """
        lmax = cl.lmax
        zs   = np.zeros(lmax+1)
        
        clmat = np.zeros( (lmax+1, 3, 3) ) # matrix of TEB correlations at each l.
        clmat[:,0,0] = getattr(cl, 'cltt', zs.copy())
        clmat[:,0,1] = getattr(cl, 'clte', zs.copy()); clmat[:,1,0] = clmat[:,0,1]
        clmat[:,0,2] = getattr(cl, 'cltb', zs.copy()); clmat[:,2,0] = clmat[:,0,2]
        clmat[:,1,1] = getattr(cl, 'clee', zs.copy())
        clmat[:,1,2] = getattr(cl, 'cleb', zs.copy()); clmat[:,2,1] = clmat[:,1,2]
        clmat[:,2,2] = getattr(cl, 'clbb', zs.copy())

        self.lmax  = lmax
        self.clmat = clmat

    def hashdict(self):
        """ return a dictionary uniquely associated with the contents of this clmat_teb. """
        return { 'lmax' : self.lmax,
                 'clmat': hashlib.md5( self.clmat.view(np.uint8) ).hexdigest() }

    def compatible(self, other):
        """ test whether this object and the clmat_teb object other can be added, subtracted, or multiplied. """
        return ( is_clmat_teb(other) and 
                 ( self.lmax == other.lmax ) and
                 ( self.clmat.shape == other.clmat.shape ) )

    def clone(self, lmax=None):
        if lmax == None:
            lmax = self.lmax
        ret = clmat_teb( util.dictobj( { 'lmax' : lmax } ) )
        ret.clmat[:,:,:] = self.clmat[0:lmax+1,:,:]

        return ret

    def __add__(self, other):
        if is_clmat_teb(other):
            assert( self.compatible(other) )
            ret = copy.deepcopy(self)
            ret.clmat += other.clmat
            return ret
        elif maps.is_tebfft(other):
            teb = other
            ret = teb.copy()
            ell = teb.get_ell()

            ret.tfft[:,:]  += np.interp( ell.flatten(), np.arange(0, len(self.clmat[:,0,0])), self.clmat[:,0,0], right=0 ).reshape(ell.shape)
            ret.efft[:,:]  += np.interp( ell.flatten(), np.arange(0, len(self.clmat[:,1,1])), self.clmat[:,1,1], right=0 ).reshape(ell.shape)
            ret.bfft[:,:]  += np.interp( ell.flatten(), np.arange(0, len(self.clmat[:,2,2])), self.clmat[:,2,2], right=0 ).reshape(ell.shape)

            return ret
        else:
            assert(0)

    def __mul__(self, other):
        if False:
            pass
        elif np.isscalar(other):
            ret = self.clone()
            ret.clmat *= other
            return ret
        elif is_clmat_teb(other):
            assert( self.compatible(other) )
            ret = self.clone()
            ret.clmat *= other.clmat
            return ret
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            lmax = self.lmax
            assert( (lmax+1) >= len(other) )

            ret = self.clone()
            for i in xrange(0,3):
                for j in xrange(0,3):
                    ret.clmat[:,i,j] *= other[0:lmax+1]

            return ret

        elif ( hasattr(other, 'tfft') and hasattr(other, 'efft') and hasattr(other, 'bfft') and hasattr(other, 'get_ell') ):
            teb = other
            ret = teb.copy()
            ell = teb.get_ell()
        
            def fftxcl(fft, cl):
                return fft * np.interp( ell.flatten(), np.arange(0, len(cl)), cl, right=0 ).reshape(fft.shape)
        
            ret.tfft[:,:]  = fftxcl( teb.tfft, self.clmat[:,0,0] ) + fftxcl( teb.efft, self.clmat[:,0,1] ) + fftxcl( teb.bfft, self.clmat[:,0,2] )
            ret.efft[:,:]  = fftxcl( teb.tfft, self.clmat[:,1,0] ) + fftxcl( teb.efft, self.clmat[:,1,1] ) + fftxcl( teb.bfft, self.clmat[:,1,2] )
            ret.bfft[:,:]  = fftxcl( teb.tfft, self.clmat[:,2,0] ) + fftxcl( teb.efft, self.clmat[:,2,1] ) + fftxcl( teb.bfft, self.clmat[:,2,2] )
        
            return ret
        else:
            assert(0)
        
    def inverse(self):
        """ return a new clmat_teb object, containing the 3x3 matrix pseudo-inverse of this one, multipole-by-multipole. """
        ret = copy.deepcopy(self)
        for l in xrange(0, self.lmax+1):
            ret.clmat[l,:,:] = np.linalg.pinv( self.clmat[l] )
        return ret

    def cholesky(self):
        """ return a new clmat_teb object, containing the 3x3 cholesky decomposition (or matrix square root) of this one, multipole-by-multipole. """
        ret = copy.deepcopy(self)
        for l in xrange(0, self.lmax+1):
            u, t, v = np.linalg.svd(self.clmat[l])
            ret.clmat[l,:,:] = np.dot(u, np.dot(np.diag(np.sqrt(t)), v))
        return ret

class blmat_teb(clmat_teb):
    """ special case of clmat_teb which is diagonal, with the TT, EE, and BB covariances all equal to b(l).
    this is a helper class for beam and pixelization transfer functions. """
    def __init__(self, bl):
        super(blmat_teb, self).__init__( util.dictobj( { 'lmax' : len(bl)-1,
                                                         'cltt' : bl,
                                                         'clee' : bl,
                                                         'clbb' : bl } ) )

def rcfft2cl( lbins, r1, r2=None, t=None, psimin=0., psimax=np.inf, psispin=1 ):
    """ calculate the annulus-averaged auto- or cross-spectrum of rfft or cfft object(s),
    for bins described by lbins and a weight function w=w(l).
            * lbins          = list of bin edges.
            * r1             = rfft or cfft object.
            * (optional) r2  = second rfft or cfft object to cross-correlate with r1 (must be same type as r1. defaults to r1, returning auto-spectrum).
            * (optional) w   = function w(l) which weights the FFT when averaging. defaults to w(l) = 1.
            * (optional) psimin, psimax, psispin = parameters used to set wedges for the annular average, only including
                   psi = mod(psispin * arctan2(lx, -ly), 2pi) in the range [psimin, psimax].
    """
    if r2 is None:
        r2 = r1
    assert( r1.compatible( r2 ) )
    
    dopsi = ( (psimin, psimax, psispin) != (0., np.inf, 1) )

    ell = r1.get_ell().flatten()

    if dopsi:
        lx, ly = r1.get_lxly()
        psi = np.mod( psispin*np.arctan2(lx, -ly), 2.*np.pi ).flatten()

    wvec = np.ones(ell.shape)
        
    if t == None:
        tvec = np.ones( ell.shape )
    else:
        tvec = t(ell)

    cvec = (r1.fft * np.conj(r2.fft)).flatten()
    wvec[ np.isnan(cvec) ] = 0.0
    cvec[ np.isnan(cvec) ] = 0.0

    if dopsi:
        w[ np.where( psi < psimin ) ] = 0.0
        w[ np.where( psi >= psimax ) ] = 0.0

    norm, bins = np.histogram(ell, bins=lbins, weights=wvec); norm[ np.where(norm != 0.0) ] = 1./norm[ np.where(norm != 0.0) ]
    clrr, bins = np.histogram(ell, bins=lbins, weights=tvec*cvec*wvec); clrr *= norm

    return bcl(lbins, { 'cl' : clrr } )

def tebfft2cl( lbins, teb1, teb2=None, t=None,  psimin=0., psimax=np.inf, psispin=1  ):
    """ calculate the annulus-averaged auto- or cross-spectrum of tebfft object(s),
    for bins described by lbins and a weight function w=w(l).
            * lbins           = list of bin edges.
            * teb1            = tebfft object.
            * (optional) teb2 = second tebfft object to cross-correlate with teb1 (defaults to teb1, returning auto-spectrum).
            * (optional) t    = function t(l) which scales FFT when averaging. defaults to t(l)=1.
            * (optional) psimin, psimax, psispin = parameters used to set wedges for the annular average, only including
                   psi = mod(psispin * arctan2(lx, -ly), 2pi) in the range [psimin, psimax].
    """
    if teb2 == None:
        teb2 = teb1
    assert( teb1.compatible( teb2 ) )
        
    dopsi = ( (psimin, psimax, psispin) != (0., np.inf, 1) )

    ell = teb1.get_ell().flatten()
    
    if dopsi:
        lx, ly = teb1.get_lxly()
        psi = np.mod( psispin*np.arctan2(lx, -ly), 2.*np.pi ).flatten()

    wvec = np.ones(ell.shape)
        
    if t == None:
        tvec = np.ones(ell.shape)
    else:
        tvec = t(ell)

    if dopsi:
        wvec[ np.where( psi < psimin ) ] = 0.0
        wvec[ np.where( psi >= psimax ) ] = 0.0

    norm, bins = np.histogram(ell, bins=lbins, weights=wvec); norm[ np.where(norm != 0.0) ] = 1./norm[ np.where(norm != 0.0) ]
    cltt, bins = np.histogram(ell, bins=lbins, weights=wvec*tvec*(teb1.tfft * np.conj(teb2.tfft)).flatten().real); cltt *= norm
    clte, bins = np.histogram(ell, bins=lbins, weights=wvec*tvec*(teb1.tfft * np.conj(teb2.efft)).flatten().real); clte *= norm
    cltb, bins = np.histogram(ell, bins=lbins, weights=wvec*tvec*(teb1.tfft * np.conj(teb2.bfft)).flatten().real); cltb *= norm
    clee, bins = np.histogram(ell, bins=lbins, weights=wvec*tvec*(teb1.efft * np.conj(teb2.efft)).flatten().real); clee *= norm
    cleb, bins = np.histogram(ell, bins=lbins, weights=wvec*tvec*(teb1.efft * np.conj(teb2.bfft)).flatten().real); cleb *= norm
    clbb, bins = np.histogram(ell, bins=lbins, weights=wvec*tvec*(teb1.bfft * np.conj(teb2.bfft)).flatten().real); clbb *= norm

    return bcl(lbins, { 'cltt' : cltt,
                        'clte' : clte,
                        'cltb' : cltb,
                        'clee' : clee,
                        'cleb' : cleb,
                        'clbb' : clbb } )

def cross_cl( lbins, r1, r2=None, w=None , t=None):
    """ returns the auto- or cross-spectra of either rfft or tebfft objects. this is a convenience wrapper around tebfft2cl and rcfft2cl. """
    if r2 is None:
        r2 = r1
    assert( r1.compatible( r2 ) )

    if maps.is_tebfft(r1):
        return tebfft2cl(lbins, r1, r2, t=t)
    elif maps.is_rfft(r1):
        return rcfft2cl(lbins, r1, r2, t=t)
    elif maps.is_cfft(r1):
        return rcfft2cl(lbins, r1, r2, t=t)
    else:
        assert(0)

def cl2cfft(cl, pix):
    """ returns a maps.cfft object with the pixelization pix, with FFT(lx,ly) = linear interpolation of cl[l] at l = sqrt(lx**2 + ly**2). """
    ell = pix.get_ell().flatten()
    
    ret = maps.cfft( nx=pix.nx, dx=pix.dx,
                     fft=np.array( np.interp( ell, np.arange(0, len(cl)), cl, right=0 ).reshape(pix.nx, pix.ny), dtype=np.complex ),
                     ny=pix.ny, dy=pix.dy )

    return ret

def cl2tebfft(cl, pix):
    """ returns a maps.tebfft object with the pixelization pix and [T,E,B]FFT(lx,ly) = linear interpolation of cl.cltt[l], cl.clee[l], cl.clbb[l] at l = sqrt(lx**2 + ly**2). """
    tebfft = maps.tebfft( nx=pix.nx, dx=pix.dx, ny=pix.ny, dy=pix.dy )
    ell = tebfft.get_ell().flatten()
    
    tebfft.tfft = np.array( np.interp( ell, np.arange(0, cl.lmax+1), cl.cltt, right=0 ).reshape(tebfft.tfft.shape), dtype=np.complex )
    tebfft.efft = np.array( np.interp( ell, np.arange(0, cl.lmax+1), cl.clee, right=0 ).reshape(tebfft.efft.shape), dtype=np.complex )
    tebfft.bfft = np.array( np.interp( ell, np.arange(0, cl.lmax+1), cl.clbb, right=0 ).reshape(tebfft.bfft.shape), dtype=np.complex )

    return maps.tebfft( nx=pix.nx, dx=pix.dx, ffts=[tebfft.tfft, tebfft.efft, tebfft.bfft], ny=pix.ny, dy=pix.dy )

def plot_cfft_cl2d( cfft, cfft2=None, smth=0, lcnt=None, cm=pl.cm.jet, t = lambda l, v : np.log(np.abs(v)), axlab=True, vmin=None, vmax=None, cbar=False):
    """ plot the two-dimensional auto- or cross- power of a cfft object.
          * cfft              = cfft object for auto- or cross-spectrum.
          * (optional) cfft2  = second cfft object for cross-correlation (defaults to power spectrum, with cfft1).
          * (optional) smth   = gaussian smoothing (in units of pixels) to apply to the 2D spectrum when plotting.
          * (optional) lcnt   = list of L contours to overplot.
          * (optional) cm     = matplotlib colormap object to use.
          * (optional) t      = scaling to apply to each mode as a function of (l).
          * (optional) axlab  = add lx and ly axis labels? (boolean).
          * (optional) vmin   = color scale minimum (float).
          * (optional) vmax   = color scale maximum (float).
          * (optional) cbar   = include a colorbar or not (boolean).
    """
    if cfft2 is None:
        cfft2 = cfft
    assert( cfft.compatible(cfft2) )

    lx, ly = cfft.get_lxly()
    
    lx     = np.fft.fftshift(lx)
    ly     = np.fft.fftshift(ly)

    ell    = np.sqrt(lx**2 + ly**2)
    
    ext    = [lx[0,0], lx[-1,-1], ly[-1,-1], ly[0,0]]

    import scipy.ndimage                    
    pl.imshow( scipy.ndimage.gaussian_filter( t(ell, np.fft.fftshift( (cfft.fft * np.conj(cfft2.fft)).real ) ), smth),
               interpolation='nearest', extent=ext, cmap=cm, vmin=vmin, vmax=vmax )
    if cbar == True:
        pl.colorbar()
    pl.contour( lx, ly, ell, levels=lcnt, colors='k', linestyles='--' )

    if axlab == True:
        pl.xlabel(r'$\ell_{x}$')
        pl.ylabel(r'$\ell_{y}$')
