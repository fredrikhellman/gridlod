import numpy as np
import copy

def linearpIndexBasis(N):
    """Compute basis b to convert from d-dimensional indices to linear indices.

    Example for d=3:
    
    b = linearpIndexBasis(NWorld)
    ind = np.dot(b, [1,2,3])

    ind contains the linear index for point (1,2,3).
    """
    cp = np.cumprod(N+1)
    b = np.hstack([[1], cp[:-1]])
    return b

def interiorpIndexMap(N):
    """Compute indices (linear order) of all interior points."""
    preIndexMap = lowerLeftpIndexMap(N-2, N)
    indexMap = np.sum(linearpIndexBasis(N))+preIndexMap
    return indexMap

def boundarypIndexMap(N):
    Np = np.prod(N+1)
    return boundarypIndexMapLarge(N)

def boundarypIndexMapSmall(N):
    Np = np.prod(N+1)
    return np.setdiff1d(np.arange(Np), interiorpIndexMap(N))
    
def boundarypIndexMapLarge(N):
    d = np.size(N)
    b = linearpIndexBasis(N)
    allRanges = [np.arange(Ni+1) for Ni in N]
    allIndices = np.array([], dtype='int64')
    for k in range(d):
        kRange = copy.copy(allRanges)
        kRange[k] = [0, N[k]]
        twoSides = np.meshgrid(*kRange)
        twoSidesIndices = reduce(np.add, map(np.multiply, b, twoSides)).flatten()
        allIndices = np.hstack([allIndices, twoSidesIndices])
    return np.unique(allIndices)

def pIndexMap(NFrom, NTo, NStep):
    NTopBasis = linearpIndexBasis(NTo)
    NTopBasis = NStep*NTopBasis

    uLinearIndex = lambda *index: np.tensordot(NTopBasis[::-1], index, (0, 0))
    indexMap = np.fromfunction(uLinearIndex, shape=NFrom[::-1]+1, dtype='int64').flatten()
    return indexMap

def lowerLeftpIndexMap(NFrom, NTo):
    return pIndexMap(NFrom, NTo, np.ones(np.size(NFrom), dtype='int64'))

def fillpIndexMap(NCoarse, NFine):
    assert np.all(np.mod(NFine, NCoarse) == 0)
    NStep = NFine/NCoarse
    return pIndexMap(NCoarse, NFine, NStep)

def cornerIndices(N):
    return fillpIndexMap(np.ones(np.size(N), dtype='int64'), N)

def numNeighboringElements(iPatchCoarse, NPatchCoarse, NWorldCoarse):
    assert np.all(iPatchCoarse >= 0)
    assert np.all(iPatchCoarse+NPatchCoarse <= NWorldCoarse)
    
    d = np.size(NWorldCoarse)
    Np = np.prod(NPatchCoarse+1)

    def neighboringElements(*index):
        iPatchCoarseRev = iPatchCoarse[::-1].reshape([d] + [1]*d)
        NWorldCoarseRev = NWorldCoarse[::-1].reshape([d] + [1]*d)
        iWorld = iPatchCoarseRev + index
        iWorldNeg = NWorldCoarseRev - iWorld
        lowerCount = np.sum(iWorld==0, axis=0)
        upperCount = np.sum(iWorldNeg==0, axis=0)
        return 2**(d-lowerCount-upperCount)
    
    numNeighboringElements = np.fromfunction(neighboringElements, shape=NPatchCoarse[::-1]+1, dtype='int64').flatten()
    return numNeighboringElements

def pCoordinates(NWorld, iPatch=None, NPatch=None):
    if iPatch is None:
        iPatch = np.zeros_like(NWorld, dtype='int64')
    if NPatch is None:
        NPatch = NWorld
    d = np.size(iPatch)
    Np = np.prod(NPatch+1)
    p = np.empty((Np,0))
    for k in range(d):
        fk = lambda *index: index[d-k-1]
        newrow = np.fromfunction(fk, shape=NPatch[::-1]+1).flatten()
        newrow = (iPatch[k]+newrow)/NWorld[k]
        p = np.column_stack([p, newrow])
    return p
