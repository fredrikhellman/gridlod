import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import gc

import fem
import util
import linalg

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

    #projectionsList = linalg.saddleNullSpace(APatch, IPatch, bPatchList, coarseNodes)

    PPatch = fem.assembleProlongationMatrix(NPatchCoarse, NCoarseElement)
    projectionsList = linalg.saddleNullSpaceHierarchicalBasis(APatch, IPatch, PPatch, bPatchList, coarseNodes)

    #PHier = fem.assembleHierarchicalBasisMatrix(NPatchCoarse, NCoarseElement)
    #projectionsList = linalg.saddleNullSpaceGeneralBasis(APatch, IPatch, PHier, bPatchList, coarseNodes)
    
    #projectionsList = linalg.solveWithBlockDiagonalPreconditioner(APatch, IPatch, bPatchList)
    #projectionsList = linalg.schurComplementSolve(APatch, IPatch, bPatchList)
    return projectionsList

class ElementCorrector:
    def __init__(self, world, k, iElementWorldCoarse):
        self.k = k
        self.iElementWorldCoarse = iElementWorldCoarse[:]
        self.world = world

        # Compute (NPatchCoarse, iElementPatchCoarse) from (k, iElementWorldCoarse, NWorldCoarse)
        d = np.size(iElementWorldCoarse)
        NWorldCoarse = world.NWorldCoarse
        iPatchWorldCoarse = np.maximum(0, iElementWorldCoarse - k)
        iEndPatchWorldCoarse = np.minimum(NWorldCoarse - 1, iElementWorldCoarse + k) + 1
        self.NPatchCoarse = iEndPatchWorldCoarse-iPatchWorldCoarse
        self.iElementPatchCoarse = iElementWorldCoarse - iPatchWorldCoarse
        self.iPatchWorldCoarse = iPatchWorldCoarse

    class FineScaleInformation:
        def __init__(self, aPatch, correctorsList):
            self.aPatch = aPatch
            self.correctorsList = correctorsList

    class CoarseScaleInformation:
        def __init__(self, Kij, LTPrimeij):
            self.Kij = Kij
            self.LTPrimeij = LTPrimeij
    
    def computeCorrectors(self, aPatch, IPatch):
        '''Compute the fine correctors over the patch.

        Compute the correctors Q_T\lambda_i (T is given by the class instance):

        (A \nabla Q_T lambda_j, \nabla lambda_i)_{U_K(T)} = (A \nabla lambda_j, \nabla lambda_i)_{T}

        and store them in the self.fsi object, together with the extracted A|_{U_k(T)}
        '''
        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse
        d = np.size(NCoarseElement)
        
        NPatchFine = NPatchCoarse*NCoarseElement
        NtFine = np.prod(NPatchFine)
        NpFineCoarseElement = np.prod(NCoarseElement+1)
        NpCoarse = np.prod(NPatchCoarse+1)
        NpFine = np.prod(NPatchFine+1)

        assert(np.size(aPatch) == NtFine)

        ALoc = world.ALoc
        MLoc = world.MLoc

        iElementPatchCoarse = self.iElementPatchCoarse
        
        elementFinetIndexMap = util.extractElementFine(NPatchCoarse,
                                                       NCoarseElement,
                                                       iElementPatchCoarse,
                                                       extractElements=True)
        elementFinepIndexMap = util.extractElementFine(NPatchCoarse,
                                                       NCoarseElement,
                                                       iElementPatchCoarse,
                                                       extractElements=False)

        AElementFull = fem.assemblePatchMatrix(NCoarseElement, ALoc, aPatch[elementFinetIndexMap])
        APatchFull = fem.assemblePatchMatrix(NPatchFine, ALoc, aPatch)
        
        bPatchFullList = []
        localBasis = world.localBasis
        for phi in localBasis.T:
            bPatchFull = np.zeros(NpFine)
            bPatchFull[elementFinepIndexMap] = AElementFull*phi
            bPatchFullList.append(bPatchFull)

        correctorsList = ritzProjectionToFinePatch(NPatchCoarse, NCoarseElement, APatchFull,
                                                   bPatchFullList, IPatch)
        self.fsi = self.FineScaleInformation(aPatch, correctorsList)
        
    def computeCoarseQuantities(self):
        '''Compute the coarse quantities K and L for this element corrector

        Compute the tensors (T is given by the class instance):

        KTij   = (A \nabla (lambda_j - Q-T lambda_j), \nabla lambda_i)_{U_k(T)}
        LTT'ij = (A \nabla (chi_T - Q_T)lambda_j, \nabla (chi_T - Q_T) lambda_j)_{T'}

        and store them in the self.csi object. See
        notes/coarse_quantities.pdf for a description.

        '''
        assert(hasattr(self, 'fsi'))

        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse
        NPatchFine = NPatchCoarse*NCoarseElement

        NTPrime = np.prod(NPatchCoarse)
        NpPatchCoarse = np.prod(NPatchCoarse+1)
        
        d = np.size(NPatchCoarse)
        
        correctorsList = self.fsi.correctorsList
        aPatch = self.fsi.aPatch

        ALoc = world.ALoc
        localBasis = world.localBasis

        TPrimeCoarsepStartIndices = util.lowerLeftpIndexMap(NPatchCoarse-1, NPatchCoarse)
        TPrimeCoarsepIndexMap = util.lowerLeftpIndexMap(np.ones_like(NPatchCoarse), NPatchCoarse)
        
        TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
        TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)
        
        TPrimeFinepStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)
        TPrimeFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)

        patchElementIndexBasis = util.linearpIndexBasis(NPatchCoarse-1)
        TInd = np.dot(patchElementIndexBasis, self.iElementPatchCoarse)
        
        # This loop can probably be done faster than this. If a bottle-neck, fix!
        Kij = np.zeros((NpPatchCoarse, 2**d))
        LTPrimeij = np.zeros((NTPrime, 2**d, 2**d))
        for (TPrimeInd, 
             TPrimeCoarsepStartIndex, 
             TPrimeFinetStartIndex, 
             TPrimeFinepStartIndex) \
             in zip(np.arange(NTPrime),
                    TPrimeCoarsepStartIndices,
                    TPrimeFinetStartIndices,
                    TPrimeFinepStartIndices):
            
            KTPrime = fem.assemblePatchMatrix(NCoarseElement, ALoc, aPatch[TPrimeFinetStartIndex +
                                                                           TPrimeFinetIndexMap])
            P = localBasis
            Q = np.column_stack([corrector[TPrimeFinepStartIndex + TPrimeFinepIndexMap] for corrector in correctorsList])
            BTPrimeij = np.dot(Q.T, KTPrime*P)
            CTPrimeij = np.dot(Q.T, KTPrime*Q)
            sigma = TPrimeCoarsepStartIndex + TPrimeCoarsepIndexMap
            if TPrimeInd == TInd:
                Aij = np.dot(P.T, KTPrime*P)
                LTPrimeij[TPrimeInd] = CTPrimeij \
                                       - BTPrimeij \
                                       - BTPrimeij.T \
                                       + Aij
                Kij[sigma,:] += Aij - BTPrimeij
            else:
                LTPrimeij[TPrimeInd] = CTPrimeij
                Kij[sigma,:] += -BTPrimeij


        self.csi = self.CoarseScaleInformation(Kij, LTPrimeij)

    def clearFineQuantities(self):
        assert(hasattr(self, 'fsi'))
        del self.fsi
        
# def computeElementCorrectorDirichletBC(NPatchCoarse,
#                                        NCoarseElement,
#                                        iElementCoarse,
#                                        APatchFull,
#                                        AElementFull,
#                                        localBasis,
#                                        IPatch):

#     ## HALLER PA HAR
    
#     # Compute rhs
#     fineIndexBasis = util.linearpIndexBasis(NPatchFine)
#     elementFineIndex = np.dot(fineIndexBasis, iElementCoarse*NCoarseElement)
#     bFull = np.zeros(NpFine)
#     bFreeList = []
#     for phi in localBasis:
#         bFull[elementFineIndex + elementToFineIndexMap] = AElementFull*phi
#         bFreeList.append(bFull[freePatch])

#     # Prepare IPatchFree
#     IPatchFree = IPatch[:,freePatch]
#     linalg.saddleNullSpace(APatchFree, IPatchFree, bFreeList)
    
#     if IPatch is None:
#         correctorFreeList = []
#         for bFree in bFreeList:
#             correctorFree,_ = sparse.linalg.cg(APatchFree, bFree, tol=1e-9)
#             correctorFreeList.append(correctorFree)
#     else:
#         IPatchFree = IPatch[:,freePatch]
#         correctorFreeList = linalg.saddle(APatchFree, IPatchFree, bFreeList)

#     correctorFullList = []
#     for correctorFree in correctorFreeList:
#         correctorFull = np.zeros(NpFine)
#         correctorFull[freePatch] = correctorFree
#         correctorFullList.append(correctorFull)
        
#     return correctorFullList

#def computePetrovGalerkinStiffnessMatrix(NWorldCoarse,
#                                         NCoarseElement,
#                                         aFlatFine,
#                                         IElement,
#                                         k):
#    
#
#def ritzProjectionToFinePatch(gp, ICoarseElement)
