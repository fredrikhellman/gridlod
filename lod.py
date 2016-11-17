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

def computeElementCorrectorDirichletBC(NPatchCoarse,
                                       NCoarseElement,
                                       iElementCoarse,
                                       APatchFull,
                                       AElementFull,
                                       localBasis,
                                       IPatch):
    d = np.size(NPatchCoarse)
    NPatchFine = NPatchCoarse*NCoarseElement
    NpFine = np.prod(NPatchFine+1)

    elementToFineIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)
    coarseToFineIndexMap = util.fillpIndexMap(NPatchCoarse, NPatchFine)
    
    # Find patch free degrees of freedom
    freePatch = util.interiorpIndexMap(NPatchFine)
    APatchFree = APatchFull[freePatch][:,freePatch]
    coarseNodes = np.arange()

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
