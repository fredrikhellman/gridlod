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

    fixedIPatch = None
    # If IPatch is nodal interpolation, then don't use Lagrange
    # multipliers, instead remove degrees of freedom
    if IPatch.nnz == IPatch.shape[0]:
        IPatchCoo = IPatch.tocoo()
        fixedIPatch = np.sort(IPatchCoo.col)
        IPatch = None

    # Find patch free degrees of freedom
    freePatch = util.interiorpIndexMap(NPatchFine)
    if not fixedIPatch is None:
        freePatch = np.setdiff1d(freePatch, fixedIPatch)

    APatchFree = APatchFull[freePatch][:,freePatch]
        
    fineIndexBasis = util.linearpIndexBasis(NPatchFine)
    elementFineIndex = np.dot(fineIndexBasis, iElementCoarse*NCoarseElement)
    bFull = np.zeros(NpFine)
    bFreeList = []
    for phi in localBasis:
        bFull[elementFineIndex + elementToFineIndexMap] = AElementFull*phi
        bFreeList.append(bFull[freePatch])

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
