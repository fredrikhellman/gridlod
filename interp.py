import numpy as np
import scipy.sparse as sparse

from util import *
      
def nodalCoarseElementMatrix(NCoarseElement):
    NpFine = np.prod(NCoarseElement+1)
    NpCoarse = 2**np.size(NCoarseElement)
    
    ind = cornerIndices(NCoarseElement)
    rows = np.arange(np.size(ind))
    I = sparse.coo_matrix((np.ones_like(rows), (rows, ind)), shape=(NpCoarse, NpFine))
    return I

def nodalPatchMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse, NCoarseElement):
    NPatchFine = NPatchCoarse*NCoarseElement
    
    IElement = nodalCoarseElementMatrix(NCoarseElement)
    IPatch = assemblePatchInterpolationMatrix(IElement, NPatchFine, NCoarseElement)
    AvgPatch = assemblePatchNodeAveragingMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse)
    INodalPatch = AvgPatch*IPatch
    return INodalPatch
    
def assemblePatchInterpolationMatrix(IElement, NPatchFine, NCoarseElement):
    assert np.all(np.mod(NPatchFine, NCoarseElement) == 0)

    d = np.size(NPatchFine)
    
    NPatchCoarse = NPatchFine/NCoarseElement

    NpFine = np.prod(NPatchFine+1)
    NpCoarse = np.prod(NPatchCoarse+1)
    
    fineToPatchMap = lowerLeftpIndexMap(NCoarseElement, NPatchFine)
    coarseToPatchMap = lowerLeftpIndexMap(np.ones(d, dtype='int64'), NPatchCoarse)
    
    IElementCoo = IElement.tocoo()

    raisedRows = coarseToPatchMap[IElementCoo.row]
    raisedCols = fineToPatchMap[IElementCoo.col]

    pCoarseInd = lowerLeftpIndexMap(NPatchCoarse-1, NPatchCoarse)
    pFineInd = pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)

    fullRows = np.add.outer(pCoarseInd, raisedRows).flatten()
    fullCols = np.add.outer(pFineInd, raisedCols).flatten()
    fullData = np.tile(IElementCoo.data, np.size(pFineInd))

    I = sparse.csr_matrix((fullData, (fullRows, fullCols)), shape=(NpCoarse, NpFine))
    
    return I

def assemblePatchNodeAveragingMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse):
    Np = np.prod(NPatchCoarse+1)
    numNeighbors = numNeighboringElements(iPatchCoarse, NPatchCoarse, NWorldCoarse)
    return sparse.dia_matrix((1./numNeighbors, 0), shape=(Np, Np))
    
