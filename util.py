import numpy as np
import copy
from functools import reduce

def linearpIndexBasis(N):
    """Compute basis b to convert from d-dimensional indices to linear indices.

    Example for d=3:
    
    b = linearpIndexBasis(NWorld)
    ind = np.dot(b, [1,2,3])

    ind contains the linear index for point (1,2,3).
    """
    cp = np.cumprod(N+1, dtype='int64')
    b = np.hstack([[1], cp[:-1]])
    return b

def convertpLinearIndexToCoordIndex(N, ind):
    ind = np.array(ind)
    d = np.size(N)
    if ind.ndim > 0:
        m = np.size(ind)
        coord = np.zeros([d, m], dtype='int64')
    else:
        coord = np.zeros([d], dtype='int64')
    basis = linearpIndexBasis(N)
    for i in range(d-1,-1,-1):
        coord[i,...] = ind//basis[i]
        ind -= coord[i,...]*basis[i]
    assert(np.all(ind == 0))
    return coord

def convertpCoordIndexToLinearIndex(N, coord):
    basis = linearpIndexBasis(N)
    return np.dot(coord, basis)

def interiorpIndexMap(N):
    """Compute indices (linear order) of all interior points."""
    preIndexMap = lowerLeftpIndexMap(N-2, N)
    indexMap = np.sum(linearpIndexBasis(N))+preIndexMap
    return indexMap

def boundarypIndexMap(N, boundaryMap=None):
    return boundarypIndexMapLarge(N, boundaryMap)

def boundarypIndexMapLarge(N, boundaryMap=None):
    d = np.size(N)
    if boundaryMap is None:
        boundaryMap = np.ones([d,2], dtype='bool')
    b = linearpIndexBasis(N)
    allRanges = [np.arange(Ni+1) for Ni in N]
    allIndices = np.array([], dtype='int64')
    for k in range(d):
        kRange = copy.copy(allRanges)
        kRange[k] = np.array([], dtype='int64')
        if boundaryMap[k][0]:
            kRange[k] = np.append(kRange[k], 0)
        if boundaryMap[k][1]:
            kRange[k] = np.append(kRange[k], N[k])
        twoSides = np.meshgrid(*kRange)
        twoSidesIndices = reduce(np.add, list(map(np.multiply, b, twoSides))).flatten()
        allIndices = np.hstack([allIndices, twoSidesIndices])
    return np.unique(allIndices)

def extractElementFine(NCoarse,
                       NCoarseElement,
                       iElementCoarse,
                       extractElements=True):
    return extractPatchFine(NCoarse, NCoarseElement, iElementCoarse, 0*iElementCoarse+1, extractElements)

def extractPatchFine(NCoarse,
                     NCoarseElement,
                     iPatchCoarse,
                     NPatchCoarse,
                     extractElements=True):
    NFine = NCoarse*NCoarseElement
    if extractElements:
        fineIndexBasis = linearpIndexBasis(NFine-1)
        patchFineIndexStart = np.dot(fineIndexBasis, iPatchCoarse*NCoarseElement)
        patchFineIndexMap = lowerLeftpIndexMap(NPatchCoarse*NCoarseElement-1, NFine-1)
    else:
        fineIndexBasis = linearpIndexBasis(NFine)
        patchFineIndexStart = np.dot(fineIndexBasis, iPatchCoarse*NCoarseElement)
        patchFineIndexMap = lowerLeftpIndexMap(NPatchCoarse*NCoarseElement, NFine)
    return patchFineIndexStart + patchFineIndexMap

def pIndexMap(NFrom, NTo, NStep):
    """ Create a map of the point indices from one grid to another.
    
    NFrom is the grid on which the map is defined
    NTo   is the grid onto which the map ranges
    NStep is the the number of steps in the To-grid for each step in the From-grid

    Example in 1D)

    NFrom = np.array([2])
    NTo   = np.array([10])
    NStep = np.array([3])

                 v        v        v
      toIndex =  0  1  2  3  4  5  6  7  8  9  10
    fromIndex =  0        1        2
        grids    |--+--+--|--+--+--|--+--+--+--+

    | denotes the nodes of NFrom (and NTo)
    + denotes the nodes of NTo 

    pIndexMap(NFrom, Nto, NStep) returns np.array([0, 3, 6]) marked by 'v' above

    (NTo[-1] is always ignored, so in this 1D-example NTo can be set to anything)
    """
    NTopBasis = linearpIndexBasis(NTo)
    NTopBasis = NStep*NTopBasis

    uLinearIndex = lambda *index: np.tensordot(NTopBasis[::-1], index, (0, 0))
    indexMap = np.fromfunction(uLinearIndex, shape=NFrom[::-1]+1, dtype='int64').flatten()
    return indexMap

def elementpIndexMap(NTo):
    return lowerLeftpIndexMap(np.ones_like(NTo), NTo)

def lowerLeftpIndexMap(NFrom, NTo):
    return pIndexMap(NFrom, NTo, np.ones(np.size(NFrom), dtype='int64'))

def fillpIndexMap(NCoarse, NFine):
    assert np.all(np.mod(NFine, NCoarse) == 0)
    NStep = NFine//NCoarse
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

def tCoordinates(NWorld, iPatch=None, NPatch=None):
    if NPatch is None:
        NPatch = NWorld-1
    else:
        NPatch = NPatch-1

    elementSize = 1./NWorld
    p = pCoordinates(NWorld, iPatch, NPatch)
    t = p + 0.5*elementSize
    return t
                    
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
        newrow = np.fromfunction(fk, shape=NPatch[::-1]+1, dtype='int64').flatten()
        newrow = (iPatch[k]+newrow)/NWorld[k]
        p = np.column_stack([p, newrow])
    return p

def fineIndicesInPatch(NWorldCoarse, NCoarseElement, iPatchCoarse, NPatchCoarse):
    NWorldFine = NCoarseElement*NWorldCoarse

    fineNodeIndexBasis = linearpIndexBasis(NWorldFine)
    fineElementIndexBasis = linearpIndexBasis(NWorldFine-1)
    
    iPatchFine = NCoarseElement*iPatchCoarse

    patchFineNodeIndices = lowerLeftpIndexMap(NPatchCoarse*NCoarseElement, NWorldFine)
    fineNodeStartIndex = np.dot(fineNodeIndexBasis, iPatchFine)
    fineNodeIndices = fineNodeStartIndex + patchFineNodeIndices

    patchFineElementIndices = lowerLeftpIndexMap(NPatchCoarse*NCoarseElement-1, NWorldFine-1)
    fineElementStartIndex = np.dot(fineElementIndexBasis, iPatchFine)
    fineElementIndices = fineElementStartIndex + patchFineElementIndices

    return fineNodeIndices, fineElementIndices

def ignoreDuplicates(row, col, data):
    # Assumes (data, row, col) not in canonical format.
    if len(data) == 0:
        return row, col, data
    order = np.lexsort((row, col))
    row = row[order]
    col = col[order]
    data = data[order]
    unique_mask = ((row[1:] != row[:-1]) |
                   (col[1:] != col[:-1]))
    unique_mask = np.append(True, unique_mask)
    row = row[unique_mask]
    col = col[unique_mask]
    data = data[unique_mask]
    return row, col, data

