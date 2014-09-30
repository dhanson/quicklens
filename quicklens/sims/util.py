# quicklens/sims/util.py
# --
# helper routines for use with the simulation libraries.
#

import numpy as np

def hash_check(hash1, hash2, ignore=[], keychain=[]):
    """ compare two hash dictionaries, usually produced by the .hashdict() method of a library object. """
    keys1 = hash1.keys()
    keys2 = hash2.keys()

    for key in ignore:
        if key in keys1: keys1.remove(key)
        if key in keys2: keys2.remove(key)

    for key in set(keys1).union(set(keys2)):
        v1 = hash1[key]
        v2 = hash2[key]

        def hashfail(msg=None):
            print "ERROR: HASHCHECK FAIL AT KEY = " + ':'.join(keychain + [key])
            if msg != None:
                print "   ", msg
            print "   ", "V1 = ", v1
            print "   ", "V2 = ", v2
            assert(0)

        if type(v1) != type(v2):
            hashfail('UNEQUAL TYPES')
        elif type(v2) == dict:
            hash_check( v1, v2, ignore=ignore, keychain=keychain + [key] )
        elif type(v1) == np.ndarray:
            if not np.allclose(v1, v2):
                hashfail('UNEQUAL ARRAY')
        else:
            if not( v1 == v2 ):
                hashfail('UNEQUAL VALUES')
