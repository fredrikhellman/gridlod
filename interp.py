import numpy as np
import scipy.sparse as sparse

import util
import fem

def nodalCoarseElementMatrix(NCoarseElement):
    NpFine = np.prod(NCoarseElement+1)
    NpCoarse = 2**np.size(NCoarseElement)
    
    ind = util.cornerIndices(NCoarseElement)
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
    
def L2ProjectionCoarseElementMatrix(NCoarseElement):
    NpFine = np.prod(NCoarseElement+1)
    NpCoarse = 2**np.size(NCoarseElement)
    
    MLoc = fem.localMassMatrix(NCoarseElement)
    MElement = fem.assemblePatchMatrix(NCoarseElement, MLoc)
    Phi = fem.localBasis(NCoarseElement)

    PhiTM = Phi.T*MElement
    PhiTMPhi = np.dot(PhiTM, Phi)

    IDense = np.dot(np.linalg.inv(PhiTMPhi), PhiTM)
    I = sparse.coo_matrix(IDense)
    return I

def L2ProjectionPatchMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse, NCoarseElement, boundaryConditions=None):
    NPatchFine = NPatchCoarse*NCoarseElement
    
    IElement = L2ProjectionCoarseElementMatrix(NCoarseElement)
    IPatch = assemblePatchInterpolationMatrix(IElement, NPatchFine, NCoarseElement)
    AvgPatch = assemblePatchNodeAveragingMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse)
    IL2ProjectionPatch = AvgPatch*IPatch
    if boundaryConditions is not None:
        BcPatch = assemblePatchBoundaryConditionMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse, boundaryConditions)
        IL2ProjectionPatch = BcPatch*IL2ProjectionPatch
        
    return IL2ProjectionPatch

def uncoupledL2ProjectionCoarseElementMatrix(NCoarseElement):
    d = np.size(NCoarseElement)
    
    NpFine = np.prod(NCoarseElement+1)

    # First compute full element P'*M
    MLoc = fem.localMassMatrix(NCoarseElement)
    MElement = fem.assemblePatchMatrix(NCoarseElement, MLoc)
    Phi = fem.localBasis(NCoarseElement)

    PhiTM = Phi.T*MElement

    # Prepare indices to go from local to element and find all corner
    # element indices
    dOnes = np.ones_like(NCoarseElement, dtype='int64')
    loc2ElementIndexMap = util.lowerLeftpIndexMap(dOnes, NCoarseElement)
    cornerElementIndices = util.pIndexMap(dOnes, NCoarseElement, NCoarseElement-1)

    # Compute P'*MCorner for all corners
    PhiTMCornersList = []
    for i in range(2**d):
        cornerInd = cornerElementIndices[i] + loc2ElementIndexMap
        PhiTMCorner = 0*PhiTM
        PhiTMCorner[:,cornerInd] = np.dot(Phi.T[:,cornerInd], MLoc)
        PhiTMCornersList.append(PhiTMCorner)

    PhiTMAllCorners = reduce(np.add, PhiTMCornersList)

    # For each corner, compute
    #    P'*M - P'*MAllCorners + P'*MCorner
    IDense = np.zeros((2**d, NpFine))
    for i in range(2**d):
        PhiTMCorner = PhiTMCornersList[i]
        PhiTMi      = PhiTM - PhiTMAllCorners + PhiTMCorner
        PhiTMiPhi   = np.dot(PhiTMi, Phi)
        
        IiDense     = np.dot(np.linalg.inv(PhiTMiPhi), PhiTMi)
        IDense[i,:] = IiDense[i,:]
        
    I = sparse.coo_matrix(IDense)
        
    return I

def uncoupledL2ProjectionPatchMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse, NCoarseElement):
    NPatchFine = NPatchCoarse*NCoarseElement
    
    IElement = uncoupledL2ProjectionCoarseElementMatrix(NCoarseElement)
    IPatch = assemblePatchInterpolationMatrix(IElement, NPatchFine, NCoarseElement)
    AvgPatch = assemblePatchNodeAveragingMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse)
    IuncoupledL2ProjectionPatch = AvgPatch*IPatch
    return IuncoupledL2ProjectionPatch

def assemblePatchInterpolationMatrix(IElement, NPatchFine, NCoarseElement):
    assert np.all(np.mod(NPatchFine, NCoarseElement) == 0)

    d = np.size(NPatchFine)
    
    NPatchCoarse = NPatchFine/NCoarseElement

    NpFine = np.prod(NPatchFine+1)
    NpCoarse = np.prod(NPatchCoarse+1)
    
    fineToPatchMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)
    coarseToPatchMap = util.lowerLeftpIndexMap(np.ones(d, dtype='int64'), NPatchCoarse)
    
    IElementCoo = IElement.tocoo()

    raisedRows = coarseToPatchMap[IElementCoo.row]
    raisedCols = fineToPatchMap[IElementCoo.col]

    pCoarseInd = util.lowerLeftpIndexMap(NPatchCoarse-1, NPatchCoarse)
    pFineInd = util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)

    fullRows = np.add.outer(pCoarseInd, raisedRows).flatten()
    fullCols = np.add.outer(pFineInd, raisedCols).flatten()
    fullData = np.tile(IElementCoo.data, np.size(pFineInd))

    I = sparse.csr_matrix((fullData, (fullRows, fullCols)), shape=(NpCoarse, NpFine))
    
    return I

def assemblePatchNodeAveragingMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse):
    Np = np.prod(NPatchCoarse+1)
    numNeighbors = util.numNeighboringElements(iPatchCoarse, NPatchCoarse, NWorldCoarse)
    return sparse.dia_matrix((1./numNeighbors, 0), shape=(Np, Np))
    
def assemblePatchBoundaryConditionMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse, boundaryConditions):
    Np = np.prod(NPatchCoarse+1)
    d = np.size(NPatchCoarse)
    
    diag = np.ones(Np)

    # Find what patch faces are common to the world faces, and inherit
    # boundary conditions from the world for those. For the other
    # faces, all DoFs free.
    boundaryMapWorld = boundaryConditions==0

    inherit0 = iPatchCoarse==0
    inherit1 = (iPatchCoarse+NPatchCoarse)==NWorldCoarse
    
    boundaryMap = np.zeros([d, 2], dtype='bool')
    boundaryMap[inherit0,0] = boundaryMapWorld[inherit0,0]
    boundaryMap[inherit1,1] = boundaryMapWorld[inherit1,1]

    fixed = util.boundarypIndexMap(NPatchCoarse, boundaryMap)
    diag[fixed] = 0

    return sparse.dia_matrix((diag, 0), shape=(Np, Np))
