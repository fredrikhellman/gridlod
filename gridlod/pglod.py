import numpy as np
import scipy.sparse as sparse
from copy import deepcopy

from . import util
from . import fem

def assembleBasisCorrectors(world, patchT, basisCorrectorsListT):
    '''Compute the basis correctors given the elementwise basis
    correctors for each coarse element.

    '''
    NWorldCoarse = world.NWorldCoarse
    NCoarseElement = world.NCoarseElement
    NWorldFine = NWorldCoarse*NCoarseElement

    NtCoarse = np.prod(NWorldCoarse)
    NpCoarse = np.prod(NWorldCoarse+1)
    NpFine = np.prod(NWorldFine+1)

    TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)

    cols = []
    rows = []
    data = []
    for TInd in range(NtCoarse):
        basisCorrectorsList = basisCorrectorsListT[TInd]
        patch = patchT[TInd]
        
        NPatchFine = patch.NPatchCoarse*NCoarseElement
        iPatchWorldFine = patch.iPatchWorldCoarse*NCoarseElement

        patchpIndexMap = util.lowerLeftpIndexMap(NPatchFine, NWorldFine)
        patchpStartIndex = util.convertpCoordIndexToLinearIndex(NWorldFine, iPatchWorldFine)

        colsT = TpStartIndices[TInd] + TpIndexMap
        rowsT = patchpStartIndex + patchpIndexMap
        dataT = np.hstack(basisCorrectorsList)

        cols.extend(np.repeat(colsT, np.size(rowsT)))
        rows.extend(np.tile(rowsT, np.size(colsT)))
        data.extend(dataT)

    basisCorrectors = sparse.csc_matrix((data, (rows, cols)), shape=(NpFine, NpCoarse))

    return basisCorrectors
        
def assembleMsStiffnessMatrix(world, KmsijT):
    '''Compute the multiscale Petrov-Galerking stiffness matrix given
    Kmsij for each coarse element.

    '''
    NWorldCoarse = world.NWorldCoarse

    NtCoarse = np.prod(world.NWorldCoarse)
    NpCoarse = np.prod(world.NWorldCoarse+1)

    TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)

    cols = []
    rows = []
    data = []
    for TInd in range(NtCoarse):
        Kmsij = KmsijT[TInd]

        NPatchCoarse = ecT.NPatchCoarse

        patchpIndexMap = util.lowerLeftpIndexMap(NPatchCoarse, NWorldCoarse)
        patchpStartIndex = util.convertpCoordIndexToLinearIndex(NWorldCoarse, ecT.iPatchWorldCoarse)

        colsT = TpStartIndices[TInd] + TpIndexMap
        rowsT = patchpStartIndex + patchpIndexMap
        dataT = Kmsij.flatten()

        cols.extend(np.tile(colsT, np.size(rowsT)))
        rows.extend(np.repeat(rowsT, np.size(colsT)))
        data.extend(dataT)

    Kms = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

    return Kms

def assembleStiffnessMatrix(world, KijT):
    '''Compute the standard coarse stiffness matrix given Kij for each
    coarse element.

    '''
    world = self.world
    NWorldCoarse = world.NWorldCoarse

    NtCoarse = np.prod(world.NWorldCoarse)
    NpCoarse = np.prod(world.NWorldCoarse+1)

    TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)

    cols = []
    rows = []
    data = []
    for TInd in range(NtCoarse):
        Kij = KijT[TInd]

        NPatchCoarse = ecT.NPatchCoarse

        colsT = TpStartIndices[TInd] + TpIndexMap
        rowsT = TpStartIndices[TInd] + TpIndexMap
        dataT = Kij.flatten()

        cols.extend(np.tile(colsT, np.size(rowsT)))
        rows.extend(np.repeat(rowsT, np.size(colsT)))
        data.extend(dataT)

    K = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

    return K
    
def solve(world, KmsFull, bFull, boundaryConditions):
    NWorldCoarse = world.NWorldCoarse
    NpCoarse = np.prod(NWorldCoarse+1)
        
    fixed = util.boundarypIndexMap(NWorldCoarse, boundaryConditions==0)
    free  = np.setdiff1d(np.arange(NpCoarse), fixed)
        
    KmsFree = KmsFull[free][:,free]
    bFree = bFull[free]

    uFree = sparse.linalg.spsolve(KmsFree, bFree)

    uFull = np.zeros(NpCoarse)
    uFull[free] = uFree

    return uFull, uFree
