import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import gc

import fem
import util
import linalg

# Saddle point problem solvers
class nullspaceSolver:
    def __init__(self, NPatchCoarse, NCoarseElement):
        NPatchFine = NPatchCoarse*NCoarseElement
        self.coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)

    def solve(self, A, I, bList):
        return linalg.saddleNullSpace(A, I, bList, self.coarseNodes)

class nullspaceOneLevelHierarchySolver:
    def __init__(self, NPatchCoarse, NCoarseElement):
        NPatchFine = NPatchCoarse*NCoarseElement
        self.coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)
        self.PPatch = fem.assembleProlongationMatrix(NPatchCoarse, NCoarseElement)
        # It is time consuming to create PPatch. Perhaps another
        # solver of this kind can be implemented, but where PPatch is
        # implicitly defined instead...
        
    def solve(self, A, I, bList):
        return linalg.saddleNullSpaceHierarchicalBasis(A, I, self.PPatch, bList, self.coarseNodes)

class nullspaceSeveralLevelsHierarchySolver:
    def __init__(self, NPatchCoarse, NCoarseElement):
        NPatchFine = NPatchCoarse*NCoarseElement
        self.coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)
        self.PHier = fem.assembleHierarchicalBasisMatrix(NPatchCoarse, NCoarseElement)

    def solve(self, A, I, bList):
        return linalg.saddleNullSpaceGeneralBasis(A, I, self.PHier, bList, self.coarseNodes)

class blockDiagonalPreconditionerSolver:
    def __init__(self):
        pass
    
    def solve(self, A, I, bList):
        return linalg.solveWithBlockDiagonalPreconditioner(A, I, bList)

class schurComplementSolver:
    def __init__(self):
        pass
    
    def solve(self, A, I, bList):
        return linalg.schurComplementSolve(A, I, bList)

def ritzProjectionToFinePatch(NPatchCoarse,
                              NCoarseElement,
                              APatchFull,
                              bPatchFullList,
                              IPatch):

    saddleSolver = nullspaceOneLevelHierarchySolver(NPatchCoarse, NCoarseElement)
    #saddleSolver = schurComplementSolver()  # Fast for small patch problems
    
    return ritzProjectionToFinePatchWithGivenSaddleSolver(NPatchCoarse,
                                                          NCoarseElement,
                                                          APatchFull,
                                                          bPatchFullList,
                                                          IPatch,
                                                          saddleSolver)

def ritzProjectionToFinePatchWithGivenSaddleSolver(NPatchCoarse,
                                                   NCoarseElement,
                                                   APatchFull,
                                                   bPatchFullList,
                                                   IPatch,
                                                   saddleSolver):
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

    projectionsList = saddleSolver.solve(APatch, IPatch, bPatchList)

    return projectionsList

class ElementCorrector:
    def __init__(self, world, k, iElementWorldCoarse, saddleSolver=None):
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

        if saddleSolver == None:
            self._saddleSolver = nullspaceOneLevelHierarchySolver(self.NPatchCoarse,
                                                                  world.NCoarseElement)
        else:
            self._saddleSolver = saddleSolver
            
    class FineScaleInformation:
        def __init__(self, aPatch, correctorsList):
            self.aPatch = aPatch
            self.correctorsList = correctorsList

    class CoarseScaleInformation:
        def __init__(self, Kij, LTPrimeij, relA=None):
            self.Kij = Kij
            self.LTPrimeij = LTPrimeij
            self.relA = relA

    @property
    def saddleSolver(self):
        return self._saddleSolver

    @saddleSolver.setter
    def saddleSolver(self, value):
        self._saddleSolver = value
            
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

            correctorsList = ritzProjectionToFinePatchWithGivenSaddleSolver(NPatchCoarse,
                                                                            NCoarseElement,
                                                                            APatchFull,
                                                                            bPatchFullList,
                                                                            IPatch,
                                                                            self.saddleSolver)
        self.fsi = self.FineScaleInformation(aPatch, correctorsList)
        
    def computeCoarseQuantities(self, relAPatch=None):
        '''Compute the coarse quantities K and L for this element corrector

        Compute the tensors (T is given by the class instance):

        KTij   = (A \nabla (lambda_j - Q-T lambda_j), \nabla lambda_i)_{U_k(T)}
        LTT'ij = (A \nabla (chi_T - Q_T)lambda_j, \nabla (chi_T - Q_T) lambda_j)_{T'}

        and store them in the self.csi object. See
        notes/coarse_quantities.pdf for a description.

        The argument relAPatch is for providing a coarse (patch)
        element relative coefficient factor, i.e. if aPatch is
        computed from a base coefficient baseA elementwise as such:

           aPatch|_T = baseA|_T*relAPatch|_T   (note relAPatch|_T is constant)

        If relAPatch is provided, relAPatch will be stored and can be used for
        efficiently computing deltaTPrime.
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


        self.csi = self.CoarseScaleInformation(Kij, LTPrimeij, relAPatch)

    def clearFineQuantities(self):
        assert(hasattr(self, 'fsi'))
        del self.fsi

    def computeErrorIndicator(self, aNewPatch, relANewPatch=None):
        assert(hasattr(self, 'csi'))
        
        world = self.world
        NPatchCoarse = self.NPatchCoarse
        NCoarseElement = world.NCoarseElement
        iElementPatchCoarse = self.iElementPatchCoarse
        NPatchFine = NCoarseElement*NPatchCoarse

        # Sanity checks
        assert(relANewPatch is not None or hasattr(self, 'fsi'))
        assert(relANewPatch is None or self.csi.relA is not None)

        ALoc = world.ALoc
        localBasis = world.localBasis
        
        elementFinetIndexMap = util.extractElementFine(NPatchCoarse,
                                                       NCoarseElement,
                                                       iElementPatchCoarse,
                                                       extractElements=True)
        
        AElement = fem.assemblePatchMatrix(NCoarseElement, ALoc, self.fsi.aPatch[elementFinetIndexMap])
        Mij = np.dot(localBasis.T, AElement*localBasis)
        LTPrimeij = self.csi.LTPrimeij
        
        relA = self.csi.relA
        
        NTPrime = np.prod(NPatchCoarse)

        if relANewPatch is not None:
            deltaMaxNormTPrime = np.abs((relaANewPatch - relA)/np.sqrt(relaANewPatch*relaA))
        else:
            deltaMaxNormTPrime = np.zeros(NTPrime)
            TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
            TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

        
        muTPrime = np.zeros(NTPrime)
        for TPrimeInd in np.arange(NTPrime):
            # Solve eigenvalue problem LTPrimeij x = mu_TPrime Mij x
            eigenvalues = scipy.linalg.eigvals(LTPrimeij[TPrimeInd][:-1,:-1], Mij[:-1,:-1])
            muTPrime[TPrimeInd] = np.max(np.real(eigenvalues))

            if relANewPatch is None:
                aPatchTPrime = self.fsi.aPatch[TPrimeFinetStartIndices[TPrimeInd] + TPrimeFinetIndexMap]
                aNewPatchTPrime = aNewPatch[TPrimeFinetStartIndices[TPrimeInd] + TPrimeFinetIndexMap]
                deltaMaxNormTPrime[TPrimeInd] = np.max(np.abs((aPatchTPrime - aNewPatchTPrime)/
                                                              np.sqrt(aPatchTPrime*aNewPatchTPrime)))

        np.set_printoptions(linewidth=220)
        print ''
        print np.reshape(muTPrime, [7,7])
        print np.reshape(deltaMaxNormTPrime, [7,7])
        pass
        
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
