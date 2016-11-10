import numpy as np
import scipy.sparse as sparse

from util import *

def passTwoBinaryIndices(f):
    return lambda *ind: f(np.array(ind[:len(ind)/2], dtype='bool'),
                          np.array(ind[len(ind)/2:], dtype='bool'))

def localMassMatrix(N):
    d = np.size(N)
    detJ = np.prod(1./N)

    def massMatrixBinaryIndices(ib, jb):
        return detJ*(1 << np.sum(~(ib ^ jb), axis=0))/6.**d

    MBin = np.fromfunction(passTwoBinaryIndices(massMatrixBinaryIndices), shape=[2]*(2*d))
    MFlat = MBin.flatten('F')
    M = MFlat.reshape(2**d, 2**d, order='F')
    return M
    
def localStiffnessMatrix(N):
    d = np.size(N)
    detJ = np.prod(1./N)

    def stiffnessMatrixBinaryIndices(ib, jb):
        M = detJ*(1 << np.sum(~(ib ^ jb), axis=0))/6.**d
        A = M*np.sum(map(np.multiply, N**2, 3*(1-3*(ib ^ jb))), axis=0)
        return A

    ABin = np.fromfunction(passTwoBinaryIndices(stiffnessMatrixBinaryIndices), shape=[2]*(2*d))
    AFlat = ABin.flatten('F')
    A = AFlat.reshape(2**d, 2**d, order='F')
    return A

def localBoundaryMatrix(N):
    d = np.size(N)
    detJd = np.prod(1./N[:-1])

    def boundaryMatrixBinaryIndices(ib, jb):
        C = detJd*(1 << np.sum(~(ib[:d-1] ^ jb[:d-1]), axis=0))/6.**(d-1)
        C *= N[d-1]*(1-2*jb[d-1])*(1-ib[d-1])
        return C

    CBin = np.fromfunction(passTwoBinaryIndices(boundaryMatrixBinaryIndices), shape=[2]*(2*d))
    CFlat = CBin.flatten('F')
    C = CFlat.reshape(2**d, 2**d, order='F')
    return C

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

def localBasis(N):
    d = np.size(N)

    Phis = [1]
    for k in range(d): 
        x = np.linspace(0,1,N[k]+1)
        newPhis0 = []
        newPhis1 = []
        for Phi in Phis:
            newPhis0.append(np.kron(1-x, Phi))
            newPhis1.append(np.kron(x, Phi))
        Phis = newPhis0 + newPhis1
    return Phis
