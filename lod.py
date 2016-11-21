import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

import fem
import util
import linalg

#def computeElementCorrector(NPatchCoarse,
#                            NCoarseElement,
#                            iElementCoarse,
#                            aFlatPatchFine,
#                            IElement):
#    ALoc = gridlod.fem.localStiffnessMatrix(NPatchFine)
#    APatchFull = fem.assemblePatchMatrix(NPatchFine, ALoc, aFlatPatchFine)
#    pass

def ritzProjectionToFinePatch(NPatchCoarse,
                              NCoarseElement,
                              APatchFull,
                              bPatchFullList,
                              IPatch):
    d = np.size(NPatchCoarse)
    NPatchFine = NPatchCoarse*NCoarseElement
    NpFine = np.prod(NPatchFine+1)

    fixed = util.boundarypIndexMap(NPatchFine)
    def imposeBoundaryConditionsStronglyOnMatrix(A, fixed):
        AStrong = A.copy()
        nzFixedCols = AStrong[:,fixed].nonzero()
        AStrong[nzFixedCols[0],fixed[nzFixedCols[1]]] = 0
        nzFixedRows = AStrong[fixed,:].nonzero()
        AStrong[fixed[nzFixedRows[0]],nzFixedRows[1]] = 0
        AStrong[fixed,fixed] = 1
        return AStrong

    def imposeBoundaryConditionsStronglyOnVector(b, fixed):
        bStrong = b.copy()
        bStrong[fixed] = 0
        return bStrong

    APatch = imposeBoundaryConditionsStronglyOnMatrix(APatchFull, fixed)
    bPatchList = [imposeBoundaryConditionsStronglyOnVector(bPatchFull, fixed) for bPatchFull in bPatchFullList]

    coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)

    projectionsList = linalg.saddleNullSpace(APatch, IPatch, bPatchList, coarseNodes)

    return projectionsList

def computeElementCorrectorDirichletBC(NPatchCoarse,
                                       NCoarseElement,
                                       iElementCoarse,
                                       APatchFull,
                                       AElementFull,
                                       localBasis,
                                       IPatch):

    ## HALLER PA HAR
    
    # Compute rhs
    fineIndexBasis = util.linearpIndexBasis(NPatchFine)
    elementFineIndex = np.dot(fineIndexBasis, iElementCoarse*NCoarseElement)
    bFull = np.zeros(NpFine)
    bFreeList = []
    for phi in localBasis:
        bFull[elementFineIndex + elementToFineIndexMap] = AElementFull*phi
        bFreeList.append(bFull[freePatch])

    # Prepare IPatchFree
    IPatchFree = IPatch[:,freePatch]
    linalg.saddleNullSpace(APatchFree, IPatchFree, bFreeList)
    
    if IPatch is None:
        correctorFreeList = []
        for bFree in bFreeList:
            correctorFree,_ = sparse.linalg.cg(APatchFree, bFree, tol=1e-9)
            correctorFreeList.append(correctorFree)
    else:
        IPatchFree = IPatch[:,freePatch]
        correctorFreeList = linalg.saddle(APatchFree, IPatchFree, bFreeList)

    correctorFullList = []
    for correctorFree in correctorFreeList:
        correctorFull = np.zeros(NpFine)
        correctorFull[freePatch] = correctorFree
        correctorFullList.append(correctorFull)
        
    return correctorFullList

#def computePetrovGalerkinStiffnessMatrix(NWorldCoarse,
#                                         NCoarseElement,
#                                         aFlatFine,
#                                         IElement,
#                                         k):
#    
#
#def ritzProjectionToFinePatch(gp, ICoarseElement)
