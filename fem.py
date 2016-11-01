import numpy as np
import scipy.sparse as sparse

from util import *

def localMassMatrix(N):
    d = np.size(N)
    detJ = np.prod(1./N)
    M = np.zeros((2**d, 2**d))
    for i in range(2**d):
        for j in range(2**d):
            delta = i^j
            factor = 1.
            for k in range(d):
                factor *= 1. + (((delta>>k) & 1)^1)
            M[i,j] = factor
    M = detJ*M/(6.**d)
    return M
    
def localStiffnessMatrix(N):
    d = np.size(N)
    A = localMassMatrix(N)

    for i in range(2**d):
        for j in range(2**d):
            s = 0
            for k in range(d):
                factor = (N[k])**2
                if ((i^j)>>k) & 1 == 0:
                    factor *= 3
                else:
                    factor *= -6
                s += factor
            A[i,j] *= s
    return A

def assemblePatchMatrix(NPatch, ALoc=None, aPatch=None):
    d = np.size(NPatch)
    Np = np.prod(NPatch+1)
    Nt = np.prod(NPatch)
    
    if aPatch is None:
        aPatch = np.ones(Nt)

    loc2Patch = lowerLeftpIndexMap(np.ones(d, 'int64'), NPatch)
    pInd = lowerLeftpIndexMap(NPatch-1, NPatch)

    loc2PatchRep = np.repeat(loc2Patch, 2**d)
    loc2PatchTile = np.tile(loc2Patch, 2**d)
    indexMatrixRows = np.add.outer(pInd, loc2PatchRep)
    indexMatrixCols = np.add.outer(pInd, loc2PatchTile)

    rows = indexMatrixRows.flatten()
    cols = indexMatrixCols.flatten()
    values = np.kron(aPatch, ALoc.flatten())

    APatch = sparse.csc_matrix((values, (rows, cols)), shape=(Np, Np))
    
    return APatch

