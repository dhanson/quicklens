# quicklens/util.py
# --
# this module contains various helper functions.

import sys, time, copy, os, io
import numpy as np

def chunks(l,n):
	""" divide list l into smaller chunks of size <= n. """
	return [l[i:i+n] for i in range(0, len(l), n)]

def enumerate_progress(list, label='', clear=False):
    """ implementation of python's enumerate built-in which
    prints a progress bar as it yields elements of a list.

    (optional) label = short info to print while running which describes the process being counted.
    (optional) clear = clear line once completed.
    """
    t0 = time.time()
    ni = len(list)

    class printer:
        def __init__(self):
            self.stdout  = sys.stdout
            sys.stdout   = self
            self.cpct    = 0

        def __del(self):
            self.stop()

        def stop(self):
            sys.stdout   = self.stdout

        def write(self,text):
            if (text != "\n"):
                self.stdout.write("\r")
                self.stdout.write("\033[K")
                self.stdout.write(text)
            else:
                self.stdout.write(text)
                self.write_bar()
                self.flush()

        def write_bar(self, cpct=None):
            if cpct == None:
                cpct = self.cpct
            self.cpct = cpct

            dt = time.time() - t0
            dh = np.floor( dt / 3600. )
            dm = np.floor( np.mod(dt, 3600.) / 60.)
            ds = np.floor( np.mod(dt, 60) )

            self.stdout.write( "\r [" + ('%02d:%02d:%02d' % (dh, dm, ds)) + "] " + label + " " + int(10. * cpct / 100)*"-" + "> " + ("%02d" % cpct) + r"%" )

        def flush(self):
            self.stdout.flush()

    printer = printer()

    for i, v in enumerate(list):
        yield i, v
        ppct = int(100. * (i-1) / ni)
        cpct = int(100. * (i+0) / ni)
        if cpct > ppct:
            printer.write_bar(cpct)

    printer.stop()

    if clear == True:
        sys.stdout.write("\r"); sys.stdout.write("\033[K"); sys.stdout.flush()
    else:
        sys.stdout.write("\n"); sys.stdout.flush()

class dictobj(object):
    """ wrapper for a dictionary, which makes its values accessible as attributes for each key. """
    def __init__(self, d):
        self.d = d

    def __getattr__(self, m):
        if m in self.d.keys():
            return self.d.get(m)
        else:
            raise AttributeError

class sum:
    """ helper class to contain the sum of a number of objects. """
    
    def __init__(self, clone=copy.deepcopy):
        self.__dict__['__clone'] = clone
        self.__dict__['__sum'] = None

    def __iadd__(self, obj):
        self.add(obj)
        return self

    def __getattr__(self, attr):
        return getattr(self.__dict__['__sum'], attr)

    def __setattr__(self, attr, val):
        setattr(self.__dict__['__sum'], attr, val)

    def add(self, obj):
        if self.__dict__['__sum'] is None:
            self.__dict__['__sum'] = self.__dict__['__clone'](obj)
        else:
            self.__dict__['__sum'] += obj

    def get(self):
        return self.__dict__['__sum']

class avg:
    """ helper class to contain the average of a number of objets. """
    
    def __init__(self, clone=copy.deepcopy):
        self.__dict__['__clone'] = clone
        self.__dict__['__sum'] = None
        self.__dict__['__num'] = 0

    def __iadd__(self, obj):
        self.add(obj)
        return self

    def __getattr__(self, attr):
        return getattr(self.get(), attr)

    def add(self, obj):
        if self.__dict__['__sum'] is None:
            self.__dict__['__sum'] = self.__dict__['__clone'](obj)
        else:
            self.__dict__['__sum'] += obj

        self.__dict__['__num'] += 1

    def get(self):
        return self.__dict__['__sum'] / self.__dict__['__num']

class dt():
    """ helper class to contain / print a time difference. """
    
    def __init__(self, _dt):
        self.dt = _dt

    def __str__(self):
        return ('%02d:%02d:%02d' % (np.floor(self.dt / 60 / 60),
                                 np.floor(np.mod(self.dt, 60*60) / 60 ),
                                 np.floor(np.mod(self.dt, 60)) ) )
    def __int__(self):
        return int(self.dt)

class stopwatch():
    """ simple stopwatch timer class. """
    
    def __init__(self):
        """ initialize the watch with start given by current time. """
        self.st = time.time()
        self.lt = self.st

    def lap(self):
        """ return a tuple containing the time since start and the time since last call to lap or elapsed. """
        lt      = time.time()
        ret     = ( dt(lt - self.st), dt(lt - self.lt) )
        self.lt = lt
        return ret

    def elapsed(self):
        """ return the time since initialization. """
        lt      = time.time()
        ret     = dt(lt - self.st)
        self.lt = lt
        return ret

class jit:
    """ just-in-time instantiator class. """
    def __init__(self, ctype, *cargs, **ckwds):
        self.__dict__['__jit_args'] = [ctype, cargs, ckwds]
        self.__dict__['__jit_obj']  = None

    def instantiate(self):
        [ctype, cargs, ckwds] = self.__dict__['__jit_args']
        print 'jit: instantiating ctype =', ctype
        self.__dict__['__jit_obj'] = ctype( *cargs, **ckwds )
        del self.__dict__['__jit_args']

    def __getattr__(self, attr):
        if self.__dict__['__jit_obj'] == None:
            self.instantiate()
        return getattr(self.__dict__['__jit_obj'], attr)

    def __setattr__(self, attr, val):
        if self.__dict__['__jit_obj'] == None:
            self.instantiate()
        setattr(self.__dict__['__jit_obj'], attr, val)

def det_3x3(weight):
    """ returns the NxM matrix containing the determinants of a set of 3x3 weight matrices. weight should have shape of the form (0:N, 0:M, 0:3, 0:3). """
    import scipy.weave as weave

    ret = np.zeros(weight[:,:,0,0].shape)
    n_pix_y, n_pix_x = weight[:,:,0,0].shape

    zero_det = np.array([0])

    c_code = """
    #line 11 "util.py"
    long double cof[3][3]; // A temporary variable

    for (int iy=0; iy<int(n_pix_y); ++iy) {
        for (int ix=0; ix<int(n_pix_x); ++ix) {
                // Take the inverse of the weight matrix in this pixel.
                cof[0][0] = weight(iy,ix,1,1)*weight(iy,ix,2,2)-weight(iy,ix,1,2)*weight(iy,ix,2,1);
                cof[1][0] = weight(iy,ix,1,2)*weight(iy,ix,2,0)-weight(iy,ix,1,0)*weight(iy,ix,2,2);
                cof[2][0] = weight(iy,ix,1,0)*weight(iy,ix,2,1)-weight(iy,ix,1,1)*weight(iy,ix,2,0);
                double det = (weight(iy,ix,0,0)*cof[0][0] + weight(iy,ix,0,1)*cof[1][0] + weight(iy,ix,0,2)*cof[2][0]);
                ret(iy,ix) = det;
            } // end for (loop over ix)
        } // end for (loop over iy)
        """
    weave.inline(c_code, ['weight', 'zero_det', 'n_pix_y', 'n_pix_x', 'ret'], type_converters=weave.converters.blitz)

    return ret

class log(object):
    """ helper class which prints log messages to stdout as well as dumping to tfname. """
    
    def __init__(self, tfname):
        self.terminal = sys.stdout

        if not os.path.exists( os.path.dirname(tfname) ):
            os.makedirs(os.path.dirname(tfname))
        self.log = open(tfname, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()

class memoize(object):
    """ a simple memoize decorator (http://en.wikipedia.org/wiki/Memoization) """
    def __init__(self, func):
        self.func = func
        self.cache = {}
    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            v = self.func(*args)
            self.cache[args] = v
            return v

# some helper functions for storing np arrays in an sqlite3 database.
try:
    import sqlite3

    def adapt_array(arr):
        out = io.BytesIO(); np.save(out, arr); out.seek(0); return buffer(out.read())
    sqlite3.register_adapter(np.ndarray, adapt_array)
    def convert_array(text):
        out = io.BytesIO(text); out.seek(0); return np.load(out)
    sqlite3.register_converter("array", convert_array)
    
except ImportError, exc:
    sys.stderr.write("IMPORT ERROR: " + __file__ + " (" + str(exc) + ")\n")

class npdb():
    """ a simple wrapper class to store numpy arrays in an sqlite3 database, indexed by an id string. """
    def __init__(self, fname):
        import mpi
        if (not os.path.exists(fname) and mpi.rank == 0):
            con = sqlite3.connect(fname, detect_types=sqlite3.PARSE_DECLTYPES)
            cur = con.cursor()
            cur.execute("CREATE TABLE npdb (id STRING PRIMARY KEY, arr ARRAY)")
            con.commit()
        mpi.barrier()
        
        self.con = sqlite3.connect(fname, timeout=60., detect_types=sqlite3.PARSE_DECLTYPES)

    def add(self, id, vec):
        assert( self.get(id) is None )
        cur = self.con.cursor()
        cur.execute("INSERT INTO npdb (id,  arr) VALUES (?,?)", (id, vec.reshape((1,len(vec)))))
        self.con.commit()
        cur.close()

    def get(self, id):
        cur = self.con.cursor()
        cur.execute("SELECT arr FROM npdb WHERE id=?", (id,))
        data = cur.fetchone()
        cur.close()
        if data is None:
            return None
        else:
            return data[0].flatten()
