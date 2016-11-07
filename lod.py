import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

import fem
import util

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
                                       IElement):
    d = np.size(NPatchCoarse)
    NPatchFine = NPatchCoarse*NCoarseElement
    NpFine = np.prod(NPatchFine+1)

    elementToFineIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)

    fixedIPatch = None
    # If IElement is nodal interpolation, then don't use Lagrange
    # multipliers, instead remove degrees of freedom
    if IElement.nnz == IElement.shape[0]:
        fixedILocal = elementToFineIndexMap[IElement.col]
        fixedIPatch = np.add.outer(util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement),
                                   fixedILocal).flatten()
        fixedIPatch = np.sort(fixedIPatch)
    else:
        raise('Lagrange multipliers not yet implemented')

    # Find patch free degrees of freedom
    freePatch = util.interiorpIndexMap(NPatchFine)
    if not fixedIPatch is None:
        freePatch = np.setdiff1d(freePatch, fixedIPatch)

    APatchFree = APatchFull[freePatch][:,freePatch]
        
    fineIndexBasis = util.linearpIndexBasis(NPatchFine)
    elementFineIndex = np.dot(fineIndexBasis, iElementCoarse*NCoarseElement)
    bFull = np.zeros(NpFine)
    correctorsFull = []
    for i in range(len(localBasis)):
        bFull[elementFineIndex + elementToFineIndexMap] = AElementFull*localBasis[i]
        bFree = bFull[freePatch]
        correctorFree,_ = sparse.linalg.cg(APatchFree, bFree, tol=1e-4)
        correctorFull = np.zeros(NpFine)
        correctorFull[freePatch] = correctorFree
        correctorsFull.append(correctorFull)
    return correctorsFull

#def computePetrovGalerkinStiffnessMatrix(NWorldCoarse,
#                                         NCoarseElement,
#                                         aFlatFine,
#                                         IElement,
#                                         k):
#    
#
#def ritzProjectionToFinePatch(gp, ICoarseElement)
