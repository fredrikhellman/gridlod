import numpy as np
import scipy.sparse as sparse

import util

def localMatrix(d, matrixValuesTakesBinaryIndices):
    def convertToBinaryIndices(f):
        return lambda *ind: f(np.array(ind[:d], dtype='bool'),
                              np.array(ind[d:], dtype='bool'))

    ABin = np.fromfunction(convertToBinaryIndices(matrixValuesTakesBinaryIndices), shape=[2]*(2*d))
    AFlat = ABin.flatten('F')
    A = AFlat.reshape(2**d, 2**d, order='F')
    return A
    

def localMassMatrix(N):
    d = np.size(N)
    detJ = np.prod(1./N)
    def massMatrixBinaryIndices(ib, jb):
        return detJ*(1 << np.sum(~(ib ^ jb), axis=0))/6.**d
    
    return localMatrix(d, massMatrixBinaryIndices)
    
def localStiffnessMatrix(N):
    d = np.size(N)
    detJ = np.prod(1./N)
    def stiffnessMatrixBinaryIndices(ib, jb):
        M = detJ*(1 << np.sum(~(ib ^ jb), axis=0))/6.**d
        A = M*np.sum(map(np.multiply, N**2, 3*(1-3*(ib ^ jb))), axis=0)
        return A
    
    return localMatrix(d, stiffnessMatrixBinaryIndices)

def localBoundaryNormalDerivativeMatrixGetter(N):
    return lambda k, neg: localBoundaryNormalDerivativeMatrix(N, k, neg)

def localBoundaryNormalDerivativeMatrix(N, k=0, neg=False):
    d = np.size(N)
    notk = np.ones_like(N,dtype='bool')
    notk[k] = False
    detJk = np.prod(1./N[notk])
    def boundaryNormalDerivativeMatrixBinaryIndices(ib, jb):
        C = detJk*(1 << np.sum(~(ib[notk] ^ jb[notk]), axis=0))/6.**(d-1)
        C *= N[k]*(1-2*(jb[k]^neg))*(1-(ib[k]^neg))
        return C
    
    return localMatrix(d, boundaryNormalDerivativeMatrixBinaryIndices)

def localBoundaryMassMatrixGetter(N):
    return lambda k, neg: localBoundaryMassMatrix(N, k, neg)

def localBoundaryMassMatrix(N, k=0, neg=False):
    d = np.size(N)
    notk = np.ones_like(N,dtype='bool')
    notk[k] = False
    detJk = np.prod(1./N[notk])
    def boundaryMassMatrixBinaryIndices(ib, jb):
        C = detJk*(1 << np.sum(~(ib[notk] ^ jb[notk]), axis=0))/6.**(d-1)
        C *= (1-(ib[k]^neg))*(1-(jb[k]^neg))
        return C
    
    return localMatrix(d, boundaryMassMatrixBinaryIndices)

def localToPatchSparsityPattern(NPatch, NSubPatch=None):
    if NSubPatch is None:
        NSubPatch = NPatch
        
    d = np.size(NPatch)
    
    loc2Patch = util.lowerLeftpIndexMap(np.ones(d, 'int64'), NPatch)
    pInd = util.lowerLeftpIndexMap(NSubPatch-1, NPatch)

    loc2PatchRep = np.repeat(loc2Patch, 2**d)
    loc2PatchTile = np.tile(loc2Patch, 2**d)
    indexMatrixRows = np.add.outer(pInd, loc2PatchRep)
    indexMatrixCols = np.add.outer(pInd, loc2PatchTile)

    rows = indexMatrixRows.flatten()
    cols = indexMatrixCols.flatten()

    return rows, cols

def assemblePatchMatrix(NPatch, ALoc, aPatch=None):
    d = np.size(NPatch)
    Np = np.prod(NPatch+1)
    Nt = np.prod(NPatch)
    
    if aPatch is None:
        aPatch = np.ones(Nt)

    rows, cols = localToPatchSparsityPattern(NPatch)
    values = np.kron(aPatch, ALoc.flatten())

    APatch = sparse.csc_matrix((values, (rows, cols)), shape=(Np, Np))
    
    return APatch

def assemblePatchBoundaryMatrix(NPatch, CLocGetter, aPatch=None):
    # Integral over part of boundary can be implemented by adding an
    # input "chi" as an indicator function to be callable with edge
    # midpoints as inputs.
    d = np.size(NPatch)
    Np = np.prod(NPatch+1)
    Nt = np.prod(NPatch)

    if aPatch is None:
        aPatch = np.ones(Nt)

    rows = []
    cols = []
    values = []
    # Loop through each dimension
    for k in range(d):
        NEdge = NPatch.copy()
        NEdge[k] = 1
        
        edgeElementInd0 = util.lowerLeftpIndexMap(NEdge-1, NPatch-1)
        rows0, cols0 = localToPatchSparsityPattern(NPatch, NSubPatch=NEdge)

        for neg in [False, True]:
            CLoc = CLocGetter(k, neg)
            if not neg:
                edgeElementIndneg = edgeElementInd0
                rowsneg = rows0
                colsneg = cols0
            else:
                pointIndexDisplacement = int(np.prod(NPatch[:k]+1)*(NPatch[k]-1))
                elementIndexDisplacement = int(np.prod(NPatch[:k])*(NPatch[k]-1))
                edgeElementIndneg = edgeElementInd0 + elementIndexDisplacement
                rowsneg = rows0 + pointIndexDisplacement
                colsneg = cols0 + pointIndexDisplacement

            valuesneg = np.kron(aPatch[edgeElementIndneg], CLoc.flatten())
            
            rows = np.hstack([rows, rowsneg])
            cols = np.hstack([cols, colsneg])
            values = np.hstack([values, valuesneg])


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
    return np.column_stack(Phis)

def assembleProlongationMatrix(NPatchCoarse, NCoarseElement): #, localBasis):
    d = np.size(NPatchCoarse)
    Phi = localBasis(NCoarseElement)
    assert np.size(Phi, 1) == 2**d

    NPatchFine = NPatchCoarse*NCoarseElement
    NtCoarse = np.prod(NPatchCoarse)
    NpCoarse = np.prod(NPatchCoarse+1)
    NpFine = np.prod(NPatchFine+1)

    rowsBasis = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)
    colsBasis = np.zeros_like(rowsBasis)
    
    rowsElement = np.tile(rowsBasis, 2**d)
    colsElement = np.add.outer(util.lowerLeftpIndexMap(np.ones(d, dtype='int64'), NPatchCoarse), colsBasis).flatten()

    rowsOffset = util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)
    colsOffset = util.lowerLeftpIndexMap(NPatchCoarse-1, NPatchCoarse)

    rows = np.add.outer(rowsOffset, rowsElement).flatten()
    cols = np.add.outer(colsOffset, colsElement).flatten()
    values = np.tile(Phi.flatten('F'), NtCoarse)

    # Remove duplicates. Slow?
    triples = dict(zip(zip(rows,cols),values))
    rows = np.array([key[0] for key in triples.keys()])
    cols = np.array([key[1] for key in triples.keys()])
    values = np.array(triples.values())
                   
    PPatch = sparse.csc_matrix((values, (rows, cols)), shape=(NpFine, NpCoarse))
    PPatch.eliminate_zeros()
    
    return PPatch
