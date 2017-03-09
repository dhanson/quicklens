# quicklens/maps.py
# --
# this module contains classes and subroutines
# for working with flat-sky temperature and polarization maps,
# as well as their 2D fourier transforms.
# overview of classes:
#     * pix       = descriptor class for a map pixelization with rectangular pixels.
#     * rmap      = real-valued map class.
#     * tqumap    = container for three real-valued maps (corresponding to temperature T, Q and U polarization).
#     * tqumap_wt = container for a 3x3 weight (or covariance matrix) object for T, Q, U.
#     * rfft      = fourier transform of an rmap.
#     * cfft      = fourier transform of a complex-valued map.
#     * tebfft    = fourier transform of a tqumap, divided into temperature T and E/B-mode polarization.

import hashlib
import numpy as np

import spec
import util

class pix(object):
    def __init__(self, nx, dx, ny=None, dy=None):
        if ny is None:
            ny = nx
        if dy is None:
            dy = dx
            
        self.nx = nx; self.ny = ny; self.dx = dx; self.dy = dy

    def hashdict(self):
        return { 'nx'   : self.nx,
                 'dx'   : self.dx,
                 'ny'   : self.ny,
                 'dy'   : self.dy }

    def __eq__(self, other):
        return self.compatible(other)

    def compatible(self, other):
        return ( (self.nx == other.nx) and
                 (self.ny == other.ny) and
                 (self.dx == other.dx) and
                 (self.dy == other.dy) )

def is_rmap(obj):
    """ ducktyping check of whether an object is an rmap. """
    return ( hasattr(obj, 'nx') and hasattr(obj, 'dx') and
             hasattr(obj, 'ny') and hasattr(obj, 'dy') and
             hasattr(obj, 'map') )

class rmap(pix):
    def __init__(self, nx, dx, map=None, ny=None, dy=None):
        """ class which contains a real-valued map """
        super( rmap, self ).__init__(nx, dx, ny=ny, dy=dy)
        if map is None:
            self.map = np.zeros( (self.ny, self.nx) )
        else:
            self.map = map
        assert( (self.ny, self.nx) == self.map.shape )

    def hashdict(self):
        """ returns a dictionary which should uniquely characterize the contents of this object """
        return { 'pix'  : super(rmap, self).hashdict(),
                 'map'  : hashlib.sha1(self.map.view(np.uint8)).hexdigest() }
        
    def copy(self):
        return rmap( self.nx, self.dx,
                     self.map.copy(),
                     ny = self.ny, dy = self.dy )

    def pad(self, nxp, nyp):
        """ make a new map with dimensions nxp (>nx), nyp (>ny) with this map at its center. """
        assert( nxp > self.nx )
        assert( nyp > self.ny )
        assert( np.mod( nxp - self.nx, 2 ) == 0 )
        assert( np.mod( nyp - self.ny, 2 ) == 0 )

        ret = rmap( nx=nxp, dx=self.dx, ny=nyp, dy=self.dy )
        ret.map[ (nyp-self.ny)/2:(nyp+self.ny)/2, (nxp-self.nx)/2:(nxp+self.nx)/2 ] = self.map
        return ret

    def compatible(self, other):
        """ check whether this map can be added, subtracted, etc. to the map 'other'. """
        return ( hasattr(other, 'map') and
                 super(rmap, self).compatible(other) )

    def get_rfft(self):
        """ return an rfft object containing the real fourier transform of this map. """
        ret = rfft( self.nx, self.dx, 
                    ny = self.ny, dy = self.dy )
        
        tfac = np.sqrt((self.dx * self.dy) / (self.nx * self.ny))
        ret.fft[:] = np.fft.rfft2(self.map) * tfac
        return ret

    def get_cfft(self):
        """ return a cfft object containing the full fourier transform of this map. """
        return self.get_rfft().get_cfft()

    def degrade(self, fac, intensive=False):
        """ degrade the size/resolution of this map by fac in each dimension. """
        assert( np.mod(self.nx, fac) == 0 )
        assert( np.mod(self.ny, fac) == 0 )

        ret = rmap( self.nx/fac, self.dx*fac, ny=self.ny/fac, dy=self.dy*fac )
        
        for i in xrange(0,fac):
            for j in xrange(0, fac):
                ret.map += self.map[i::fac,j::fac]

        if intensive == True:
            ret.map *= (1./fac**2)

        return ret

    def prograde(self, fac):
        """ increase the size/resolution of this map by fac in each dimension. """
        ret = rmap( self.nx*fac, self.dx/fac, ny=self.ny*fac, dy=self.dy/fac )

        for i in xrange(0,fac):
            for j in xrange(0, fac):
                ret.map[i::fac,j::fac] = self.map

        return ret

    def __mul__(self, other):
        if False:
            pass
        elif np.isscalar(other):
            ret = self.copy()
            ret.map *= other
            return ret
        elif is_rmap(other):
            assert( self.compatible(other) )
            ret = self.copy()
            ret.map *= other.map
            return ret
        elif (getattr(other, 'shape', ()) == (self.ny, self.nx)):
            ret = self.copy()
            ret.map *= other
            return ret
        else:
            assert(0)

    def __imul__(self, other):
        if False:
            pass
        elif is_rmap(other):
            assert( self.compatible(other) )
            self.map *= other.map
            return self
        elif (getattr(other, 'shape', ()) == (self.ny, self.nx)):
            self.map *= other
            return self
        else:
            assert(0)

    def __add__(self, other):
        assert( self.compatible(other) )
        return rmap( self.nx, self.dx, self.map+other.map, ny = self.ny, dy = self.dy )

    def __sub__(self, other):
        assert( self.compatible(other) )
        return rmap( self.nx, self.dx, self.map-other.map, ny = self.ny, dy = self.dy )

    def __iadd__(self, other):
        assert( self.compatible(other) )
        self.map += other.map
        return self

    def __isub__(self, other):
        assert( self.compatible(other) )
        self.map -= other.map
        return self

    def make_apod(self, fwhm_wght=10., wdth_bord=30., fwhm_apod=15., maxthresh=None, avgthresh=None):
        """ construct an apodization mask, taking this map as a set of weights. the process is
           (1) smooth the map, with a full-width-at-half-maximum (fwhm) given by fwhm_weight (in arcmin).
           (2) threshold the smoothed weights, as a percentage of the smoothed maximum (maxthresh) and/or of the smoothed average (avgthresh).
           (3) remove all pixels within a distance of wdth_bord (in arcmin) of any pixels which have been thresholded to zero.
           (4) apply a gaussian apodization, with fwhm given by fwhm_apod. """
        import scipy.ndimage

        assert( self.dx == self.dy )
        reso_arcmin = (self.dx * 180.*60./np.pi)

        smoothwt  = scipy.ndimage.gaussian_filter( self.map, fwhm_wght / reso_arcmin / 2*np.sqrt(2*np.log(2)) )

        threshwt  = np.ones( smoothwt.shape )
        if maxthresh != None:
            threshwt[ np.where(smoothwt / smoothwt.flatten().max() < maxthresh) ] = 0.0
        if avgthresh != None:
            threshwt[ np.where(smoothwt / smoothwt.flatten()[np.where(smoothwt.flatten() != 0)].avg() < avgthresh) ] = 0.0

        npix_bord = 2.*int(wdth_bord/reso_arcmin)
        xs, ys    = np.meshgrid( np.linspace(-1., 1., npix_bord), np.linspace(-1., 1., npix_bord) )
        kern_bord = np.ones( (npix_bord, npix_bord) )
        kern_bord[ np.where( (xs**2 + ys**2) >= 1. ) ] = 0.0
        bordwt    = scipy.ndimage.minimum_filter( threshwt, footprint=kern_bord )

        return rmap( self.nx, self.dx, ny=self.ny, dy=self.dy,
                     map=scipy.ndimage.gaussian_filter( bordwt, fwhm_apod / reso_arcmin / 2*np.sqrt(2*np.log(2)) ) )

def is_tqumap(obj):
    return ( hasattr(obj, 'nx') and hasattr(obj, 'dx') and
             hasattr(obj, 'ny') and hasattr(obj, 'dy') and
             hasattr(obj, 'tmap') and hasattr(obj, 'qmap') and hasattr(obj, 'umap') )

class tqumap(pix):
    def __init__(self, nx, dx, maps=None, ny=None, dy=None):
        """ class which contains temperature (T) and polarization (Q, U) maps. """
        
        super( tqumap, self ).__init__(nx, dx, ny=ny, dy=dy)
        if maps is None:
            self.tmap = np.zeros( (self.ny, self.nx) )
            self.qmap = np.zeros( (self.ny, self.nx) )
            self.umap = np.zeros( (self.ny, self.nx) )
        else:
            [self.tmap, self.qmap, self.umap] = maps

        assert( (self.ny, self.nx) == self.tmap.shape )
        assert( (self.ny, self.nx) == self.qmap.shape )
        assert( (self.ny, self.nx) == self.umap.shape )

    def copy(self):
        return tqumap( self.nx, self.dx,
                       [self.tmap.copy(), self.qmap.copy(), self.umap.copy()],
                       ny = self.ny, dy = self.dy )

    def pad(self, nxp, nyp):
        """ make a new map with dimensions nxp (>nx), nyp (>ny) with this map at its center. """
        assert( nxp > self.nx )
        assert( nyp > self.ny )
        assert( np.mod( nxp - self.nx, 2 ) == 0 )
        assert( np.mod( nyp - self.ny, 2 ) == 0 )

        ret = tqumap( nx=nxp, dx=self.dx, ny=nyp, dy=self.dy )
        for this, that in [ [self.tmap, ret.tmap], [self.qmap, ret.qmap], [self.umap, ret.umap] ]:
            that[ (nyp-self.ny)/2:(nyp+self.ny)/2, (nxp-self.nx)/2:(nxp+self.nx)/2 ] = this
        return ret

    def threshold(self, vmin, vmax=None, vcut=0.):
        """ returns a new, thresholded version of the current map.
        threshold(v) -> set all pixels which don't satisfy (-|v| < val < |v|) equal to vcut.
        threshold(min,max) -> set all pixels which don't satisfy (vmin < val < vmax) equal to vcut.
        """
        if vmax is None:
            vmin = -np.abs(vmin)
            vmax = +np.abs(vmin)
        assert( vmin < vmax )

        ret = self.copy()
        for m in [ret.tmap, ret.qmap, ret.umap]:
            m[np.where(m < vmin)] = vcut
            m[np.where(m > vmax)] = vcut
        return ret

    def compatible(self, other):
        """ check whether this map can be added, subtracted, etc. to the map 'other'. """
        return ( hasattr(other, 'tmap') and
                 hasattr(other, 'qmap') and
                 hasattr(other, 'umap') and
                 super(tqumap, self).compatible(other) )

    def get_teb(self):
        """ return a tebfft object containing the fourier transform of the T,Q,U maps. """
        ret = tebfft( self.nx, self.dx, ny = self.ny, dy = self.dy )
        
        lx, ly = ret.get_lxly()
        tpi  = 2.*np.arctan2(lx, -ly)

        tfac = np.sqrt((self.dx * self.dy) / (self.nx * self.ny))
        qfft = np.fft.rfft2(self.qmap) * tfac
        ufft = np.fft.rfft2(self.umap) * tfac
        
        ret.tfft[:] = np.fft.rfft2(self.tmap) * tfac
        ret.efft[:] = (+np.cos(tpi) * qfft + np.sin(tpi) * ufft)
        ret.bfft[:] = (-np.sin(tpi) * qfft + np.cos(tpi) * ufft)
        return ret

    def degrade(self, fac, intensive=False):
        """ degrade the size/resolution of this map by fac in each dimension. """
        assert( np.mod(self.nx, fac) == 0 )
        assert( np.mod(self.ny, fac) == 0 )

        ret = tqumap( self.nx/fac, self.dx*fac, ny=self.ny/fac, dy=self.dy*fac )
        
        for i in xrange(0,fac):
            for j in xrange(0, fac):
                ret.tmap += self.tmap[i::fac,j::fac]
                ret.qmap += self.qmap[i::fac,j::fac]
                ret.umap += self.umap[i::fac,j::fac]

        if intensive == True:
            ret *= (1./fac**2)

        return ret

    def get_chi(self, pixel_radius=2, field='B'):
        """ estimate the \chi_E or \chi_B fields from the Q and U maps using finite differences,
        following Smith and Zaldarriaga (2006) http://arxiv.org/abs/astro-ph/0610059 """
        assert( self.dx == self.dy )

        if pixel_radius==1:
            w = [1., 1./2, 0, 0, 0, 0]
        elif pixel_radius==2:
            w = [4./3, 2./3, -1./12, -1./24, 0, 0]
        else:
            assert(0)

        w = np.asarray(w)
        w /= self.dx * self.dy

        def roll(array, shift):
            out = array
            if shift[0]:
                out = np.roll( out, shift[0], axis=1 )
            if shift[1]:
                out = np.roll( out, shift[1], axis=0 )
            return out

        if field=='B':
            q, u = self.qmap, -self.umap
        elif field=='E':
            u, q = self.qmap, -self.umap
        else:
            assert(0)

        chi= ( w[0]*( roll(u,[+1,0]) - roll(u,[0,-1]) + roll(u,[-1,0]) - roll(u,[0,+1]) )
               -w[1]*( roll(q,[-1,+1]) + roll(q,[+1,-1]) - roll(q,[+1,+1]) - roll(q,[-1,-1]) ) )
        if w[2]:
            chi += w[2]*( roll(u,[+2,0]) - roll(u,[0,-2]) + roll(u,[-2,0]) - roll(u,[0,+2]) )
        if w[3]:
            chi -= w[3]*( roll(q,[-2,+2]) + roll(q,[+2,-2]) - roll(q,[+2,+2]) - roll(q,[-2,-2]) )

        return rmap( self.nx, self.dx, chi, ny=self.ny, dy=self.dy )

    def get_t(self):
        return rmap( self.nx, self.dx, map=self.tmap, ny=self.ny, dy=self.dy )
    def get_q(self):
        return rmap( self.nx, self.dx, map=self.qmap, ny=self.ny, dy=self.dy )
    def get_u(self):
        return rmap( self.nx, self.dx, map=self.umap, ny=self.ny, dy=self.dy )

    def __mul__(self, other):
        if False:
            pass
        elif is_tqumap(other):
            assert( self.compatible(other) )
            ret = self.copy()
            ret.tmap *= other.tmap
            ret.qmap *= other.qmap
            ret.umap *= other.umap
            return ret
        elif is_tqumap_wt(other):
            return other * self
        elif (getattr(other, 'shape', ()) == (self.ny, self.nx)):
            ret = self.copy()
            ret.tmap *= other
            ret.qmap *= other
            ret.umap *= other
            return ret
        else:
            assert(0)

    def __imul__(self, other):
        if False:
            pass
        elif is_tqumap(other):
            assert( self.compatible(other) )
            self.tmap *= other.tmap
            self.qmap *= other.qmap
            self.umap *= other.umap
            return self
        elif (getattr(other, 'shape', ()) == (self.ny, self.nx)):
            self.tmap *= other
            self.qmap *= other
            self.umap *= other
            return self
        else:
            assert(0)

    def __add__(self, other):
        assert( self.compatible(other) )
        return tqumap( self.nx, self.dx,
                       [self.tmap + other.tmap, self.qmap + other.qmap, self.umap + other.umap],
                       ny = self.ny, dy = self.dy )

    def __sub__(self, other):
        assert( self.compatible(other) )
        return tqumap( self.nx, self.dx,
                       [self.tmap - other.tmap, self.qmap - other.qmap, self.umap - other.umap],
                       ny = self.ny, dy = self.dy )

    def __iadd__(self, other):
        assert( self.compatible(other) )
        self.tmap += other.tmap; self.qmap += other.qmap; self.umap += other.umap
        return self

    def __isub__(self, other):
        assert( self.compatible(other) )
        self.tmap -= other.tmap; self.qmap -= other.qmap; self.umap -= other.umap
        return self

def is_tqumap_wt(obj):
    """ ducktyping check of whether an object is an tqumap_wt. """
    return ( hasattr(obj, 'nx') and hasattr(obj, 'dx') and
             hasattr(obj, 'ny') and hasattr(obj, 'dy') and
             hasattr(obj, 'weight') )

class tqumap_wt(pix):
    def __init__(self, nx, dx, weight=None, ny=None, dy=None):
        """ class which contains a 3x3 weight or covariance matrix for each pixel of a tqumap."""
        super( tqumap_wt, self ).__init__(nx, dx, ny=ny, dy=dy)
        if weight is None:
            self.weight = np.zeros( (self.ny, self.nx, 3, 3) )
        else:
            self.weight = weight

        assert( (self.ny, self.nx, 3, 3) == self.weight.shape )

    def hashdict(self):
        return { 'pix'    : super(tqumap_wt, self).hashdict(),
                 'weight' : hashlib.sha1(self.weight.view(np.uint8)).hexdigest() }

    def __mul__(self, other):
        if False:
            pass
        elif is_tqumap(other):
            assert( self.compatible(other) )
            tqu = other
            weight = self.weight
            
            reti = tqu.tmap*weight[:,:,0,0] + tqu.qmap*weight[:,:,0,1] + tqu.umap*weight[:,:,0,2]
            retq = tqu.tmap*weight[:,:,1,0] + tqu.qmap*weight[:,:,1,1] + tqu.umap*weight[:,:,1,2]
            retu = tqu.tmap*weight[:,:,2,0] + tqu.qmap*weight[:,:,2,1] + tqu.umap*weight[:,:,2,2]
            return tqumap( tqu.nx, tqu.dx, ny=tqu.ny, dy=tqu.dy, maps=[reti, retq, retu] )
        else:
            assert(0)

def make_tqumap_wt( pix, ninv=None, mask=None, ninv_dcut=None, nlev_tp=None, maskt=None, maskq=None, masku=None ):
    """ helper function to generate a tqumap_wt which describes an inverse-noise covariance matrix.
           * pix = pixelization for the tqumap.
           * (optional) ninv      = tqumat_wt object. pixels for which this matrix weight function has determinant < ninv_dcut will be masked.
           * (optional) mask      = global mask map to apply (effectively taking noise level to infinity for pixels where mask is zero).
           * (optional) ninv_dcut = used only in conjunction with ninv.
           * (optional) nlev_tp   = a tuple (nT, nP) giving pixel temperature/polarization white noise levels to use for the noise covariance in uK.arcmin.
           * (optional) maskt, maskq, masku = individual T, Q and U masks to apply.
           """
    ret    = tqumap_wt( pix.nx, pix.dx, ny=pix.ny, dy=pix.dy )

    if ninv != None:
        assert( ret.compatible(ninv) )
        
        ret.weight[:,:,:,:] = ninv.weight[:,:,:,:]
        if ninv_dcut != None:
            dets = np.abs(util.det_3x3(ninv.weight))
    else:
        assert(ninv_dcut is None)

    if nlev_tp != None:
        ret.weight = np.zeros( ret.weight.shape )
        ret.weight[:,:,0,0] = (180.*60./np.pi)**2 * ret.dx * ret.dy / nlev_tp[0]**2
        ret.weight[:,:,1,1] = (180.*60./np.pi)**2 * ret.dx * ret.dy / nlev_tp[1]**2
        ret.weight[:,:,2,2] = (180.*60./np.pi)**2 * ret.dx * ret.dy / nlev_tp[1]**2

    for i in xrange(0,3):
        for j in xrange(0,3):
            if (ninv_dcut != None):
                print "cutting ", len( np.where(dets < ninv_dcut)[0] ), " pixels for det"
                ret.weight[:,:,i,j][np.where(dets < ninv_dcut)] = 0.0

    if mask != None:
        for i in xrange(0,3):
            for j in xrange(0,3):
                ret.weight[:,:,i,j] *= mask

    if maskt != None:
        for i in xrange(0,3):
            ret.weight[:,:,i,0] *= maskt
            ret.weight[:,:,0,i] *= maskt

    if maskq != None:
        for i in xrange(0,3):
            ret.weight[:,:,i,1] *= maskq
            ret.weight[:,:,1,i] *= maskq

    if masku != None:
        for i in xrange(0,3):
            ret.weight[:,:,i,2] *= masku
            ret.weight[:,:,2,i] *= masku
    
    return ret

def is_tebfft(obj):
    """ ducktyping check of whether an object is a tebfft. """
    return ( hasattr(obj, 'nx') and hasattr(obj, 'dx') and
             hasattr(obj, 'ny') and hasattr(obj, 'dy') and
             hasattr(obj, 'tfft') and hasattr(obj, 'efft') and hasattr(obj, 'bfft') )

class tebfft(pix):
    def __init__(self, nx, dx, ffts=None, ny=None, dy=None):
        """ class which contains the FFT of a tqumap. temperature (T), E- and B-mode polarization. """
        super( tebfft, self ).__init__(nx, dx, ny=ny, dy=dy)

        if ffts is None:
            self.tfft = np.zeros( (self.ny, self.nx/2+1), dtype=np.complex )
            self.efft = np.zeros( (self.ny, self.nx/2+1), dtype=np.complex )
            self.bfft = np.zeros( (self.ny, self.nx/2+1), dtype=np.complex )
        else:
            [self.tfft, self.efft, self.bfft] = ffts

        assert( (self.ny, self.nx/2+1) == self.tfft.shape )
        assert( (self.ny, self.nx/2+1) == self.efft.shape )
        assert( (self.ny, self.nx/2+1) == self.bfft.shape )

    def hashdict(self):
        return { 'pix'  : super(tebfft, self).hashdict(),
                 'tfft' : hashlib.sha1(self.tfft.view(np.uint8)).hexdigest(),
                 'efft' : hashlib.sha1(self.efft.view(np.uint8)).hexdigest(),
                 'bfft' : hashlib.sha1(self.bfft.view(np.uint8)).hexdigest() }

    def get_ml( self, lbins, t=None, psimin=0., psimax=np.inf, psispin=1 ):
        """" returns a Cl object containing average over rings of the FFT.
                 * lbins   = list of bin edges.
                 * t       = function t(l) which scales the FFT before averaging. defaults to unity.
                 * psimin, psimax, psispin = parameters used to set wedges for the averaging.
                         psi = mod(psispin * arctan2(lx, -ly), 2pi) in the range [psimin, psimax].
        """
        dopsi = ( (psimin, psimax, psispin) != (0., np.inf, 1) )
        
        l = self.get_ell().flatten()
        if dopsi:
            lx, ly = self.get_lxly()
            psi = np.mod( psispin*np.arctan2(lx, -ly), 2.*np.pi ).flatten()
        lb = 0.5*(lbins[:-1] + lbins[1:])
            
        if t is None:
            t = np.ones(l.shape)
        else:
            t = t(l)

        cldict = {}
        for field in ['t', 'e', 'b']:
            c = getattr(self, field + 'fft').flatten()
            m = np.ones(c.shape)
        
            m[ np.isnan(c) ] = 0.0
            c[ np.isnan(c) ] = 0.0

            if dopsi:
                m[ np.where( psi < psimin ) ] = 0.0
                m[ np.where( psi >= psimax ) ] = 0.0

            norm, bins = np.histogram(l, bins=lbins, weights=m) # get number of modes in each l-bin.
            clrr, bins = np.histogram(l, bins=lbins, weights=m*t*c) # bin the spectrum.

            # normalize the spectrum.
            clrr[np.nonzero(norm)] /= norm[np.nonzero(norm)]
            cldict['cl' + field*2] = clrr
    
        return spec.bcl(lbins, cldict )
    
    def __imul__(self, other):
        if ( np.isscalar(other) or ( (type(other) == np.ndarray) and
                                     (getattr(other, 'shape', None) == self.tfft.shape) ) ):
            self.tfft *= other
            self.efft *= other
            self.bfft *= other
            return self
        elif is_rfft(other) and pix.compatible(self, other):
            self.tfft *= other.fft
            self.efft *= other.fft
            self.bfft *= other.fft
            return self
        elif (len(getattr(other, 'shape', [])) == 1):
            tfac = np.interp( self.get_ell().flatten(), np.arange(0, len(other)), other, right=0 ).reshape((self.ny,self.nx/2+1))
            self.tfft *= tfac; self.efft *= tfac; self.bfft *= tfac
            return self
        else:
            assert(0)

    def __mul__(self, other):
        if ( np.isscalar(other) or ( (type(other) == np.ndarray) and
                                     (getattr(other, 'shape', None) == self.tfft.shape) ) ):
            return tebfft( self.nx, self.dx,
                           ffts=[self.tfft * other,
                                 self.efft * other,
                                 self.bfft * other],
                           ny=self.ny, dy=self.dy )
        elif (type(other) == np.ndarray) and (len(getattr(other, 'shape', [])) == 1):
            tfac = np.interp( self.get_ell().flatten(), np.arange(0, len(other)), other, right=0 ).reshape((self.ny,self.nx/2+1))
            return tebfft( self.nx, self.dx,
                           ffts=[self.tfft * tfac,
                                 self.efft * tfac,
                                 self.bfft * tfac],
                           ny=self.ny, dy=self.dy )
        elif is_rfft(other) and pix.compatible(self, other):
            return tebfft( self.nx, self.dx,
                           ffts=[self.tfft * other.fft,
                                 self.efft * other.fft,
                                 self.bfft * other.fft],
                           ny=self.ny, dy=self.dy )
        elif is_tebfft(other) and self.compatible(other):
            return tebfft( self.nx, self.dx,
                           ffts=[self.tfft * other.tfft,
                                 self.efft * other.efft,
                                 self.bfft * other.bfft],
                           ny=self.ny, dy=self.dy )
        elif spec.is_clmat_teb(other):
            return other * self
        elif spec.is_camb_clfile(other):
            return spec.clmat_teb(other) * self
        else:
            assert(0)

    def __div__(self, other):
        if ( np.isscalar(other) or ( (type(other) == np.ndarray) and
                                     (getattr(other, 'shape', None) == self.tfft.shape) ) ):
            return tebfft( self.nx, self.dx,
                           ffts=[self.tfft,
                                 self.efft,
                                 self.bfft],
                           ny=self.ny, dy=self.dy )
        elif is_rfft(other) and pix.compatible(self, other):
            return tebfft( self.nx, self.dx,
                           ffts=[np.nan_to_num(self.tfft / other.fft),
                                 np.nan_to_num(self.efft / other.fft),
                                 np.nan_to_num(self.bfft / other.fft)],
                           ny=self.ny, dy=self.dy )
        elif is_tebfft(other) and self.compatible(other):
            return tebfft( self.nx, self.dx,
                           ffts=[np.nan_to_num(self.tfft / other.tfft),
                                 np.nan_to_num(self.efft / other.efft),
                                 np.nan_to_num(self.bfft / other.bfft)],
                           ny=self.ny, dy=self.dy )
        else:
            assert(0)

    def __rdiv__(self, other):
        if np.isscalar(other):
            return tebfft( self.nx, self.dx,
                           ffts=[other/self.tfft,
                                 other/self.efft,
                                 other/self.bfft],
                           ny=self.ny, dy=self.dy )
        else:
            assert(0)

    def compatible(self, other):
        return ( hasattr(other, 'tfft') and
                 hasattr(other, 'efft') and
                 hasattr(other, 'bfft') and
                 super(tebfft, self).compatible(other) )

    def copy(self):
        return tebfft( self.nx, self.dx,
                       [self.tfft.copy(), self.efft.copy(), self.bfft.copy()],
                       ny = self.ny, dy = self.dy )

    def inverse(self):
        """ return a new tebfft for which all elements have been set to their inverses, with exception of zeros which are untouched. """
        tfft_inv = np.zeros(self.tfft.shape, dtype=np.complex); tfft_inv[self.tfft != 0] = 1./self.tfft[self.tfft != 0]
        efft_inv = np.zeros(self.efft.shape, dtype=np.complex); efft_inv[self.efft != 0] = 1./self.efft[self.efft != 0]
        bfft_inv = np.zeros(self.bfft.shape, dtype=np.complex); bfft_inv[self.bfft != 0] = 1./self.bfft[self.bfft != 0]

        ret = tebfft( self.nx, self.dx,
                      [tfft_inv, efft_inv, bfft_inv],
                      ny = self.ny, dy = self.dy )

        return ret

    def degrade(self, fac):
        """ reduce the resolution of this map by a factor fac. """
        assert( np.mod(self.nx, fac) == 0 )
        assert( np.mod(self.ny, fac) == 0 )
        assert( np.mod(self.nx/fac, 2) == 0 )

        return tebfft( nx=self.nx/fac, dx=self.dx*fac,
                       ffts = [ self.tfft[0:self.ny/fac,0:self.nx/fac/2+1],
                                self.efft[0:self.ny/fac,0:self.nx/fac/2+1],
                                self.bfft[0:self.ny/fac,0:self.nx/fac/2+1] ],
                       ny=self.ny/fac, dy=self.dy*fac )

    def get_pix_transf(self):
        """ return the FFT describing the map-level transfer function for the pixelization of this object. """
        return rfft( self.nx, self.dx, ny=self.ny, dy=self.dy ).get_pix_transf()

    def get_cl( self, lbins, t=None, psimin=0., psimax=np.inf, psispin=1 ):
        """ returns a Cl object containing the auto-spectra of T,E,B in this map. """
        return spec.tebfft2cl( lbins, self, t=t, psimin=psimin, psimax=psimax, psispin=psispin  )

    def get_tqu(self):
        """ returns the tqumap given by the inverse Fourier transform of this object. """
        lx, ly = self.get_lxly()
        tpi  = 2.*np.arctan2(lx, -ly)

        tfac = np.sqrt((self.nx * self.ny) / (self.dx * self.dy))
    
        tmap = np.fft.irfft2(self.tfft) * tfac
        qmap = np.fft.irfft2(np.cos(tpi)*self.efft - np.sin(tpi)*self.bfft) * tfac
        umap = np.fft.irfft2(np.sin(tpi)*self.efft + np.cos(tpi)*self.bfft) * tfac

        return tqumap( self.nx, self.dx, [tmap, qmap, umap], ny = self.ny, dy = self.dy )

    def get_ffts(self):
        """ returns a list of the individual (real) ffts for T, E, B. """
        return [ rfft( self.nx, self.dx, fft=self.tfft, ny=self.ny, dy=self.dy ),
                 rfft( self.nx, self.dx, fft=self.efft, ny=self.ny, dy=self.dy ),
                 rfft( self.nx, self.dx, fft=self.bfft, ny=self.ny, dy=self.dy ) ]

    def get_cffts(self):
        """ returns a list of the individual (complex) ffts for T, E, B. """
        return [ rfft( self.nx, self.dx, fft=self.tfft, ny=self.ny, dy=self.dy ).get_cfft(),
                 rfft( self.nx, self.dx, fft=self.efft, ny=self.ny, dy=self.dy ).get_cfft(),
                 rfft( self.nx, self.dx, fft=self.bfft, ny=self.ny, dy=self.dy ).get_cfft() ]

    def get_lxly(self):
        """ returns the (lx, ly) pair associated with each Fourier mode in T, E, B. """
        return np.meshgrid( np.fft.fftfreq( self.nx, self.dx )[0:self.nx/2+1]*2.*np.pi,
                            np.fft.fftfreq( self.ny, self.dy )*2.*np.pi )

    def get_ell(self):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode in T, E, B. """
        lx, ly = self.get_lxly()
        return np.sqrt(lx**2 + ly**2)

    def __add__(self, other):
        assert( self.compatible(other) )
        return tebfft( self.nx, self.dx,
                       [self.tfft + other.tfft, self.efft + other.efft, self.bfft + other.bfft],
                       ny = self.ny, dy = self.dy )

    def __sub__(self, other):
        assert( self.compatible(other) )
        return tebfft( self.nx, self.dx,
                       [self.tfft - other.tfft, self.efft - other.efft, self.bfft - other.bfft],
                       ny = self.ny, dy = self.dy )

    def __iadd__(self, other):
        assert( self.compatible(other) )
        self.tfft += other.tfft; self.efft += other.efft; self.bfft += other.bfft
        return self

    def __isub__(self, other):
        assert( self.compatible(other) )
        self.tfft -= other.tfft; self.efft -= other.efft; self.bfft -= other.bfft
        return self

    def get_l_masked( self, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None ):
        """ returns a copy of this object which has been masked to zero in a customizable range of Fourier space. """
        lx, ly = self.get_lxly()
        ell    = np.sqrt(lx**2 + ly**2)
        mask   = np.ones( self.tfft.shape )
        if lmin  != None: mask[ np.where(ell < lmin) ] = 0.0
        if lmax  != None: mask[ np.where(ell >=lmax) ] = 0.0
        if lxmin != None: mask[ np.where(np.abs(lx) < lxmin) ] = 0.0
        if lymin != None: mask[ np.where(np.abs(ly) < lymin) ] = 0.0
        if lxmax != None: mask[ np.where(np.abs(lx) >=lxmax) ] = 0.0
        if lymax != None: mask[ np.where(np.abs(ly) >=lymax) ] = 0.0

        return tebfft( self.nx, self.dx,
                       [self.tfft * mask, self.efft * mask, self.bfft * mask],
                       ny = self.ny, dy = self.dy )

    def get_l_mask( self, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None ):
        """ return a Fourier mask for the pixelization associated with this object which is zero over customizable ranges of L. """
        lx, ly = self.get_lxly()
        ell    = np.sqrt(lx**2 + ly**2)
        mask   = np.ones( self.tfft.shape )
        if lmin  != None: mask[ np.where(ell < lmin) ] = 0.0
        if lmax  != None: mask[ np.where(ell >=lmax) ] = 0.0
        if lxmin != None: mask[ np.where(np.abs(lx) < lxmin) ] = 0.0
        if lymin != None: mask[ np.where(np.abs(ly) < lymin) ] = 0.0
        if lxmax != None: mask[ np.where(np.abs(lx) >=lxmax) ] = 0.0
        if lymax != None: mask[ np.where(np.abs(ly) >=lymax) ] = 0.0

        return tebfft( self.nx, self.dx,
                       [mask, mask, mask],
                       ny = self.ny, dy = self.dy )

def is_rfft(obj):
    """ ducktyping check of whether an object is an rfft. """
    if not ( hasattr(obj, 'nx') and hasattr(obj, 'dx') and
             hasattr(obj, 'ny') and hasattr(obj, 'dy') and
             hasattr(obj, 'fft')  ): return False

    return obj.fft.shape == (obj.nx, obj.ny/2+1)

class rfft(pix):
    def __init__(self, nx, dx, fft=None, ny=None, dy=None):
        """ class which contains the FFT of an rmap. """
        super( rfft, self ).__init__(nx, dx, ny=ny, dy=dy)

        if fft is None:
            fft = np.zeros( (self.ny, self.nx/2+1), dtype=np.complex )
        self.fft = fft

        assert( (self.ny, self.nx/2+1) == self.fft.shape )

    def __iadd__(self, other):
        if False:
            pass
        elif is_rfft(other):
            assert( self.compatible(other) )
            self.fft[:,:] += other.fft[:,:]
            return self
        else:
            assert(0)

    def __add__(self, other):
        if False:
            pass
        elif is_rfft(other):
            assert( self.compatible(other) )

            ret = self.copy()
            ret.fft[:,:] += other.fft[:,:]
            return ret
        else:
            assert(0)

    def __sub__(self, other):
        if False:
            pass
        elif is_rfft(other):
            assert( self.compatible(other) )

            ret = self.copy()
            ret.fft[:,:] -= other.fft[:,:]
            return ret
        else:
            assert(0)

    def __div__(self, other):
        if False:
            pass
        elif is_rfft(other):
            assert( self.compatible(other) )

            ret = self.copy()
            ret.fft[:,:] /= other.fft[:,:]
            return ret
        else:
            assert(0)

    def __rdiv__(self, other):
        print "rfft rdiv, other = ", other
        if False:
            pass
        elif np.isscalar(other):
            ret = self.copy()
            ret.fft[:,:] = other / self.fft[:,:]
            return ret
        else:
            assert(0)

    def __mul__(self, other):
        if False:
            pass
        elif is_rfft(other):
            assert( self.compatible(other) )

            ret = self.copy()
            ret.fft[:,:] *= other.fft[:,:]
            return ret
        elif np.isscalar(other):
            ret = self.copy()
            ret.fft *= other
            return ret
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            ell = self.get_ell()

            ret = self.copy()
            ret.fft *= np.interp( ell.flatten(), np.arange(0, len(other)), other, right=0 ).reshape(self.fft.shape)
            return ret
        else:
            assert(0)

    def __rmul__(self, other):
        return self.__mul__(other)

    def get_pix_transf(self):
        """ return the FFT describing the map-level transfer function for the pixelization of this object. """
        lx, ly = self.get_lxly()

        fft = np.zeros( self.fft.shape )
        fft[ 0, 0] = 1.0
        fft[ 0,1:] = np.sin(self.dx*lx[ 0,1:]/2.) / (self.dx * lx[0,1:] / 2.)
        fft[1:, 0] = np.sin(self.dy*ly[1:, 0]/2.) / (self.dy * ly[1:,0] / 2.)
        fft[1:,1:] = np.sin(self.dx*lx[1:,1:]/2.) * np.sin(self.dy*ly[1:,1:]/2.) / (self.dx * self.dy * lx[1:,1:] * ly[1:,1:] / 4.)

        return rfft( self.nx, self.dx, ny=self.ny, dy=self.dy, fft=fft  )

    def compatible(self, other):
        return ( hasattr(other, 'fft') and
                 getattr(other, 'fft', np.array([])).shape == self.fft.shape and
                 super(rfft, self).compatible(other) )

    def copy(self):
        return rfft( self.nx, self.dx, self.fft.copy(), ny = self.ny, dy = self.dy )

    def get_cl( self, lbins, t=None ):
        return spec.rcfft2cl( lbins, self, t=t )

    def get_rmap( self ):
        """ return the rmap given by this FFT. """
        tfac = np.sqrt((self.nx * self.ny) / (self.dx * self.dy))
        
        return rmap( self.nx, self.dx, map=np.fft.irfft2(self.fft)*tfac, ny=self.ny, dy=self.dy )

    def get_cfft( self ):
        """ return the complex FFT. """
        fft = np.zeros( (self.ny, self.nx), dtype=np.complex )
        fft[:,0:(self.nx/2+1)] = self.fft[:,:]
        fft[0,(self.nx/2+1):]  = np.conj(self.fft[0,1:self.nx/2][::-1])
        fft[1:,(self.nx/2+1):]  = np.conj(self.fft[1:,1:self.nx/2][::-1,::-1])

        return cfft( self.nx, self.dx, fft=fft, ny=self.ny, dy=self.dy )

    def get_lxly(self):
        """ returns the (lx, ly) pair associated with each Fourier mode in T, E, B. """
        return np.meshgrid( np.fft.fftfreq( self.nx, self.dx )[0:self.nx/2+1]*2.*np.pi,
                            np.fft.fftfreq( self.ny, self.dy )*2.*np.pi )

    def get_ell(self):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode in T, E, B. """
        lx, ly = self.get_lxly()
        return np.sqrt(lx**2 + ly**2)

def is_cfft(obj):
    """ ducktyping check of whether an object is a cfft. """
    return (hasattr(obj, 'nx') and hasattr(obj, 'ny') and hasattr(obj, 'dx') and hasattr(obj, 'dy') and hasattr(obj, 'fft'))

class cfft(pix):
    """
    Complex FFT object.
       fft,         numpy complex ndarray, containing the fft
       nx,          number of pixels in the x direction
       dx,          size of pixels in the x direction [units of radians]
    """
    def __init__(self, nx, dx, fft=None, ny=None, dy=None):
        super( cfft, self ).__init__(nx, dx, ny=ny, dy=dy)

        if fft is None:
            fft = np.zeros( (self.ny, self.nx), dtype=np.complex )
        self.fft = fft

        assert( (self.ny, self.nx) == self.fft.shape )

    def hashdict(self):
        """ returns a dictionary which should uniquely characterize the contents of this object """
        return { 'pix'  : super(cfft, self).hashdict(),
                 'fft'  : hashlib.sha1(self.fft.view(np.uint8)).hexdigest() }

    def __mul__(self, other):
        if False:
            pass
        elif ( hasattr(other, 'nx') and hasattr(other, 'nx') and
               hasattr(other, 'dx') and hasattr(other, 'dx') and hasattr(other, 'fft') ):
            assert( self.compatible(other) )

            ret = self.copy()
            ret.fft[:,:] = self.fft[:,:] * other.fft[:,:]
            return ret
        elif np.isscalar(other):
            ret = self.copy()
            ret.fft *= other
            return ret
            
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            ell = self.get_ell()

            ret = self.copy()
            ret.fft *= np.interp( ell.flatten(), np.arange(0, len(other)), other, right=0 ).reshape(self.fft.shape)
            return ret
        else:
            assert(0)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __div__(self, other):
        if False:
            pass
        elif is_cfft(other):
            assert( self.compatible(other) )

            ret = self.copy()
            ret.fft[:,:] = self.fft[:,:] / other.fft[:,:]
            return ret
        elif np.isscalar(other):
            ret = self.copy()
            ret.fft /= other
            return ret
        
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            ell = self.get_ell()

            ret = self.copy()
            ret.fft /= np.interp( ell.flatten(), np.arange(0, len(other)), other, right=0 ).reshape(self.fft.shape)
            return ret
        else:
            assert(0)

    def __rdiv__(self, other):
        if False:
            pass
        elif np.isscalar(other):
            ret = self.copy()
            ret.fft = other/ret.fft
            return ret
        else:
            assert(0)

    def __add__(self, other):
        if False:
            pass
        elif is_cfft(other):
            assert( self.compatible(other) )
            
            ret = self.copy()
            ret.fft += other.fft
            return ret
        elif ( ( getattr(other, 'size', 0) > 1 ) and ( len( getattr(other, 'shape', ()) ) == 1 ) ):
            ell = self.get_ell()

            ret = self.copy()
            ret.fft += np.interp( ell.flatten(), np.arange(0, len(other)), other, right=0 ).reshape(self.fft.shape)
            return ret
        else:
            assert(0)

    def __sub__(self, other):
        if False:
            pass
        elif ( hasattr(other, 'nx') and hasattr(other, 'nx') and
               hasattr(other, 'dx') and hasattr(other, 'dx') and hasattr(other, 'fft') ):
            assert( self.compatible(other) )

            return cfft( nx=self.nx, dx=self.dx, ny=self.ny, dy=self.dy, fft = (self.fft - other.fft) )
        else:
            assert(0)

    def __pow__(self, p2):
        ret = self.copy()
        ret.fft = self.fft**p2
        return ret
            
    def compatible(self, other):
        """ check whether this map can be added, subtracted, etc. to the map 'other'. """
        return ( hasattr(other, 'fft') and
                 getattr(other, 'fft', np.array([])).shape == self.fft.shape,
                 super(cfft, self).compatible(other) )

    def copy(self):
        """ return a clone of this cfft. """
        return cfft( self.nx, self.dx, self.fft.copy(), ny = self.ny, dy = self.dy )

    def conj(self):
        """ return a new cfft which is the complex conjugate of this one. """
        ret = self.copy()
        ret.fft = np.conj(ret.fft)
        return ret

    def get_cl( self, lbins, t=None ):
        """ returns a Cl object containing the auto-spectra of this map. """
        return spec.rcfft2cl( lbins, self, t=t )

    def get_ml( self, lbins, t=None, psimin=0., psimax=np.inf, psispin=1 ):
        """" returns a Cl object containing average over rings of the FFT.
                 * lbins   = list of bin edges.
                 * t       = function t(l) which weights the FFT before averaging. defaults to unity.
                 * psimin, psimax, psispin = parameters used to set wedges for the averaging.
                         psi = mod(psispin * arctan2(lx, -ly), 2pi) in the range [psimin, psimax].
        """
        dopsi = ( (psimin, psimax, psispin) != (0., np.inf, 1) )
        
        l = self.get_ell().flatten()
        if dopsi:
            lx, ly = self.get_lxly()
            psi = np.mod( psispin*np.arctan2(lx, -ly), 2.*np.pi ).flatten()
        lb = 0.5*(lbins[:-1] + lbins[1:])
            
        if t is None:
            t = np.ones(l.shape)
        else:
            t = t(l)
        
        c = self.fft.flatten()
        m = np.ones(c.shape)
        
        m[ np.isnan(c) ] = 0.0
        c[ np.isnan(c) ] = 0.0

        if dopsi:
            m[ np.where( psi < psimin ) ] = 0.0
            m[ np.where( psi >= psimax ) ] = 0.0

        norm, bins = np.histogram(l, bins=lbins, weights=m) # get number of modes in each l-bin.
        clrr, bins = np.histogram(l, bins=lbins, weights=m*t*c) # bin the spectrum.

        # normalize the spectrum.
        clrr[np.nonzero(norm)] /= norm[np.nonzero(norm)]
    
        return spec.bcl(lbins, { 'cl' : clrr } )

    def get_lxly(self):
        """ returns the (lx, ly) pair associated with each Fourier mode. """
        return np.meshgrid( np.fft.fftfreq( self.nx, self.dx )*2.*np.pi,
                            np.fft.fftfreq( self.ny, self.dy )*2.*np.pi )

    def get_ell(self):
        """ returns the wavenumber l = \sqrt(lx**2 + ly**2) for each Fourier mode """
        lx, ly = self.get_lxly()
        return np.sqrt(lx**2 + ly**2)

    def get_pix_transf(self):
        """ return the FFT describing the map-level transfer function for the pixelization of this object. """
        lx, ly = self.get_lxly()

        fft = np.zeros( self.fft.shape )
        fft[0,  0] = 1.0
        fft[0, 1:] = np.sin(self.dx*lx[ 0,1:]/2.) / (self.dx * lx[0,1:] / 2.)
        fft[1:, 0] = np.sin(self.dy*ly[1:, 0]/2.) / (self.dy * ly[1:,0] / 2.)
        fft[1:,1:] = np.sin(self.dx*lx[1:,1:]/2.) * np.sin(self.dy*ly[1:,1:]/2.) / (self.dx * self.dy * lx[1:,1:] * ly[1:,1:] / 4.)

        return cfft( self.nx, self.dx, ny=self.ny, dy=self.dy, fft=fft )

    def get_rffts( self ):
        """ return the real-valued FFT objects corresponding to the real and imaginary parts of the map associated with this fft. """
        cmap = np.fft.ifft2( self.fft )

        return ( rfft( self.nx, self.dx, fft=np.fft.rfft2(cmap.real), ny=self.ny, dy=self.dy ),
                 rfft( self.nx, self.dx, fft=np.fft.rfft2(cmap.imag), ny=self.ny, dy=self.dy ) )

    def get_l_masked( self, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None ):
        """ returns a copy of this object which has been masked to zero in a customizable range of Fourier space. """
        ret = self.copy()
        lx, ly = ret.get_lxly()
        ell    = np.sqrt(lx**2 + ly**2)
        if lmin  != None: ret.fft[ np.where(ell < lmin) ] = 0.0
        if lmax  != None: ret.fft[ np.where(ell >=lmax) ] = 0.0
        if lxmin != None: ret.fft[ np.where(np.abs(lx) < lxmin) ] = 0.0
        if lymin != None: ret.fft[ np.where(np.abs(ly) < lymin) ] = 0.0
        if lxmax != None: ret.fft[ np.where(np.abs(lx) >=lxmax) ] = 0.0
        if lymax != None: ret.fft[ np.where(np.abs(ly) >=lymax) ] = 0.0
        return ret

    def get_l_mask( self, lmin=None, lmax=None, lxmin=None, lxmax=None, lymin=None, lymax=None ):
        """ return a Fourier mask for the pixelization associated with this object which is zero over customizable ranges of L. """
        ret = self.copy()
        ret.fft[:,:] = 1.0
        lx, ly = ret.get_lxly()
        ell    = np.sqrt(lx**2 + ly**2)
        if lmin  != None: ret.fft[ np.where(ell < lmin) ] = 0.0
        if lmax  != None: ret.fft[ np.where(ell >=lmax) ] = 0.0
        if lxmin != None: ret.fft[ np.where(np.abs(lx) < lxmin) ] = 0.0
        if lymin != None: ret.fft[ np.where(np.abs(ly) < lymin) ] = 0.0
        if lxmax != None: ret.fft[ np.where(np.abs(lx) >=lxmax) ] = 0.0
        if lymax != None: ret.fft[ np.where(np.abs(ly) >=lymax) ] = 0.0
        return ret
