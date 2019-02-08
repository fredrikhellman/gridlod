import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import gc
import warnings

from . import fem
from . import util
from . import linalg
from . import coef
from . import transport

# Saddle point problem solvers
class NullspaceSolver:
    def __init__(self, NPatchCoarse, NCoarseElement):
        NPatchFine = NPatchCoarse*NCoarseElement
        self.coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)

    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        raise NotImplementedError
        return linalg.saddleNullSpace(A, I, bList, self.coarseNodes)

class NullspaceOneLevelHierarchySolver:
    def __init__(self, NPatchCoarse, NCoarseElement):
        NPatchFine = NPatchCoarse*NCoarseElement
        self.coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)
        self.PPatch = fem.assembleProlongationMatrix(NPatchCoarse, NCoarseElement)
        # It is time consuming to create PPatch. Perhaps another
        # solver of this kind can be implemented, but where PPatch is
        # implicitly defined instead...
        
    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        return linalg.saddleNullSpaceHierarchicalBasis(A, I, self.PPatch, bList, self.coarseNodes, fixed)

class NullspaceSeveralLevelsHierarchySolver:
    def __init__(self, NPatchCoarse, NCoarseElement):
        NPatchFine = NPatchCoarse*NCoarseElement
        self.coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)
        self.PHier = fem.assembleHierarchicalBasisMatrix(NPatchCoarse, NCoarseElement)

    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        raise NotImplementedError
        return linalg.saddleNullSpaceGeneralBasis(A, I, self.PHier, bList, self.coarseNodes)

class BlockDiagonalPreconditionerSolver:
    def __init__(self):
        pass
    
    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        raise NotImplementedError
        return linalg.solveWithBlockDiagonalPreconditioner(A, I, bList)

class SchurComplementSolver:
    def __init__(self, NCache=None):
        if NCache is not None:
            self.cholCache = linalg.choleskyCache(NCache)
        else:
            self.cholCache = None
    
    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        return linalg.schurComplementSolve(A, I, bList, fixed, NPatchCoarse, NCoarseElement, self.cholCache)

class DirectSolver:
    def __init__(self):
        pass
    
    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        return linalg.saddleDirect(A, I, bList, fixed)
    
def ritzProjectionToFinePatch(world,
                              iPatchWorldCoarse,
                              NPatchCoarse,
                              APatchFull,
                              bPatchFullList,
                              IPatch):

    #saddleSolver = NullspaceOneLevelHierarchySolver(NPatchCoarse, NCoarseElement)
    saddleSolver = SchurComplementSolver()  # Fast for small patch problems
    
    return ritzProjectionToFinePatchWithGivenSaddleSolver(world,
                                                          iPatchWorldCoarse,
                                                          NPatchCoarse,
                                                          APatchFull,
                                                          bPatchFullList,
                                                          IPatch,
                                                          saddleSolver)

def ritzProjectionToFinePatchWithGivenSaddleSolver(world,
                                                   iPatchWorldCoarse,
                                                   NPatchCoarse,
                                                   APatchFull,
                                                   bPatchFullList,
                                                   IPatch,
                                                   saddleSolver):
    d = np.size(NPatchCoarse)
    NPatchFine = NPatchCoarse*world.NCoarseElement
    NpFine = np.prod(NPatchFine+1)

    # Find what patch faces are common to the world faces, and inherit
    # boundary conditions from the world for those. For the other
    # faces, all DoFs fixed (Dirichlet)
    boundaryMapWorld = world.boundaryConditions==0

    inherit0 = iPatchWorldCoarse==0
    inherit1 = (iPatchWorldCoarse+NPatchCoarse)==world.NWorldCoarse
    
    boundaryMap = np.ones([d, 2], dtype='bool')
    boundaryMap[inherit0,0] = boundaryMapWorld[inherit0,0]
    boundaryMap[inherit1,1] = boundaryMapWorld[inherit1,1]

    # Using schur complement solver for the case when there are no
    # Dirichlet conditions does not work. Fix if necessary.
    assert(np.any(boundaryMap == True))
    
    fixed = util.boundarypIndexMap(NPatchFine, boundaryMap)
    
    #projectionsList = saddleSolver.solve(APatch, IPatch, bPatchList)

    projectionsList = saddleSolver.solve(APatchFull, IPatch, bPatchFullList, fixed, NPatchCoarse, world.NCoarseElement)

    return projectionsList

class FineScaleInformation:
    def __init__(self, coefficientPatch, correctorsList, IPatch):
        self.coefficient = coefficientPatch
        self.correctorsList = correctorsList
        self.IPatch = IPatch

class CoarseScaleInformation:
    def __init__(self, Kij, Kmsij, muTPrime):
        self.Kij = Kij
        self.Kmsij = Kmsij
        #self.LTPrimeij = LTPrimeij
        self.muTPrime = muTPrime
        
class ElementCorrector:
    def __init__(self, world, k, iElementWorldCoarse, IPatchGenerator, saddleSolver=None):
        self.k = k
        self.iElementWorldCoarse = iElementWorldCoarse[:]
        self.world = world
        self.IPatchGenerator = IPatchGenerator

        # Compute (NPatchCoarse, iElementPatchCoarse) from (k, iElementWorldCoarse, NWorldCoarse)
        d = np.size(iElementWorldCoarse)
        NWorldCoarse = world.NWorldCoarse
        iPatchWorldCoarse = np.maximum(0, iElementWorldCoarse - k).astype('int64')
        iEndPatchWorldCoarse = np.minimum(NWorldCoarse - 1, iElementWorldCoarse + k).astype('int64') + 1
        self.NPatchCoarse = iEndPatchWorldCoarse-iPatchWorldCoarse
        self.iElementPatchCoarse = iElementWorldCoarse - iPatchWorldCoarse
        self.iPatchWorldCoarse = iPatchWorldCoarse

        if saddleSolver == None:
            self._saddleSolver = SchurComplementSolver()
        else:
            self._saddleSolver = saddleSolver
            
        self.fsi = None
            
    @property
    def saddleSolver(self):
        return self._saddleSolver

    @saddleSolver.setter
    def saddleSolver(self, value):
        self._saddleSolver = value
            
    def computeElementCorrector(self, coefficientPatch, ARhsList=None, MRhsList=None):
        '''Compute the fine correctors over the patch.

        Compute the correctors

        (A \nabla Q_T_j, \nabla vf)_{U_K(T)} = (A \nabla ARhs_j, \nabla vf)_{T} + (MRhs_j, vf)_{T}
        '''

        assert(ARhsList is not None or MRhsList is not None)
        numRhs = None

        if ARhsList is not None:
            assert(numRhs is None or numRhs == len(ARhsList))
            numRhs = len(ARhsList)

        if MRhsList is not None:
            assert(numRhs is None or numRhs == len(MRhsList))
            numRhs = len(MRhsList)

        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse
        d = np.size(NCoarseElement)

        IPatch = self.IPatchGenerator(self.iPatchWorldCoarse, self.NPatchCoarse)
        
        NPatchFine = NPatchCoarse*NCoarseElement
        NtFine = np.prod(NPatchFine)
        NpFineCoarseElement = np.prod(NCoarseElement+1)
        NpCoarse = np.prod(NPatchCoarse+1)
        NpFine = np.prod(NPatchFine+1)

        aPatch = coefficientPatch.aFine

        assert(aPatch.shape[0] == NtFine)
        assert(aPatch.ndim == 1 or aPatch.ndim == 3)
        
        if aPatch.ndim == 1:
            ALocFine = world.ALocFine
        elif aPatch.ndim == 3:
            ALocFine = world.ALocMatrixFine
            
        MLocFine = world.MLocFine

        iElementPatchCoarse = self.iElementPatchCoarse
        elementFinetIndexMap = util.extractElementFine(NPatchCoarse,
                                                       NCoarseElement,
                                                       iElementPatchCoarse,
                                                       extractElements=True)
        elementFinepIndexMap = util.extractElementFine(NPatchCoarse,
                                                       NCoarseElement,
                                                       iElementPatchCoarse,
                                                       extractElements=False)

        if ARhsList is not None:
            AElementFull = fem.assemblePatchMatrix(NCoarseElement, ALocFine, aPatch[elementFinetIndexMap])
        if MRhsList is not None:
            MElementFull = fem.assemblePatchMatrix(NCoarseElement, MLocFine)
        APatchFull = fem.assemblePatchMatrix(NPatchFine, ALocFine, aPatch)

        bPatchFullList = []
        for rhsIndex in range(numRhs):
            bPatchFull = np.zeros(NpFine)
            if ARhsList is not None:
                bPatchFull[elementFinepIndexMap] += AElementFull*ARhsList[rhsIndex]
            if MRhsList is not None:
                bPatchFull[elementFinepIndexMap] += MElementFull*MRhsList[rhsIndex]
            bPatchFullList.append(bPatchFull)

        correctorsList = ritzProjectionToFinePatchWithGivenSaddleSolver(world,
                                                                        self.iPatchWorldCoarse,
                                                                        NPatchCoarse,
                                                                        APatchFull,
                                                                        bPatchFullList,
                                                                        IPatch,
                                                                        self.saddleSolver)
        
        self.fsi = FineScaleInformation(coefficientPatch, correctorsList, IPatch)

    def clearFineQuantities(self):
        self.fsi = None

class CoarseBasisElementCorrector(ElementCorrector):
    def __init__(self, world, k, iElementWorldCoarse, IPatchGenerator, saddleSolver=None):
        super().__init__(world, k, iElementWorldCoarse, IPatchGenerator, saddleSolver=None)
        self.csi = None
        
    def computeCorrectors(self, coefficientPatch):
        '''Compute the fine correctors over the patch.

        Compute the correctors Q_T\lambda_i (T is given by the class instance):

        (A \nabla Q_T lambda_j, \nabla vf)_{U_K(T)} = (A \nabla lambda_j, \nabla vf)_{T}

        and store them in the self.fsi object, together with the extracted A|_{U_k(T)}
        '''
        d = np.size(self.NPatchCoarse)
        ARhsList = list(map(np.squeeze, np.hsplit(self.world.localBasis, 2**d)))

        self.computeElementCorrector(coefficientPatch, ARhsList)
        
    def computeErrorIndicatorFine(self, coefficientNew):
        assert(self.fsi is not None)

        NPatchCoarse = self.NPatchCoarse
        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchFine = NPatchCoarse*NCoarseElement
        
        a = coefficientNew.aFine

        assert(a.ndim == 1 or a.ndim == 3)
        
        if a.ndim == 1:
            ALocFine = world.ALocFine
        else:
            ALocFine = world.ALocMatrixFine
        P = world.localBasis

        aTilde = self.fsi.coefficient.aFine

        TFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)
        iElementPatchFine = self.iElementPatchCoarse*NCoarseElement
        TFinetStartIndex = util.convertpCoordIndexToLinearIndex(NPatchFine-1, iElementPatchFine)

        # Compute A^-1 (A_T - A)**2. This has to be done in a batch. inv works batchwise
        if a.ndim == 1:
            b = 1./a*(aTilde-a)**2
        else:
            aInv = np.linalg.inv(a)
            b = np.einsum('Tij, Tjl, Tlk -> Tik', aInv, aTilde - a, aTilde - a)
        
        bT = b[TFinetStartIndex + TFinetIndexMap]
        PatchNorm = fem.assemblePatchMatrix(NPatchFine, ALocFine, b)
        TNorm = fem.assemblePatchMatrix(NCoarseElement, ALocFine, bT)
        
        BNorm = fem.assemblePatchMatrix(NCoarseElement, ALocFine, a[TFinetStartIndex + TFinetIndexMap])

        TFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)
        TFinepStartIndex = util.convertpCoordIndexToLinearIndex(NPatchFine, iElementPatchFine)

        Q = np.column_stack(self.fsi.correctorsList)
        QT = Q[TFinepStartIndex + TFinepIndexMap,:]

        A = np.dot((P-QT).T, TNorm*(P-QT)) + np.dot(Q.T, PatchNorm*Q)
        B = np.dot(P.T, BNorm*P)

        eigenvalues = scipy.linalg.eigvals(A[:-1,:-1], B[:-1,:-1])
        epsilonTSquare = np.max(np.real(eigenvalues))

        return np.sqrt(epsilonTSquare)

    def computeCoarseQuantities(self):
        '''Compute the coarse quantities K and L for this element corrector

        Compute the tensors (T is given by the class instance):

        KTij   = (A \nabla lambda_j, \nabla lambda_i)_{T}
        KmsTij = (A \nabla (lambda_j - Q_T lambda_j), \nabla lambda_i)_{U_k(T)}
        muTT'  = max_{w_H} || A \nabla (\chi_T - Q_T) w_H ||^2_T' / || A \nabla w_H ||^2_T

        and store them in the self.csi object. See
        notes/coarse_quantities*.pdf for a description.

        Auxiliary quantities are computed, but not saved, e.g.

        LTT'ij = (A \nabla (chi_T - Q_T)lambda_j, \nabla (chi_T - Q_T) lambda_j)_{T'}
        '''
        assert(self.fsi is not None)

        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse
        NPatchFine = NPatchCoarse*NCoarseElement

        NTPrime = np.prod(NPatchCoarse)
        NpPatchCoarse = np.prod(NPatchCoarse+1)
        
        d = np.size(NPatchCoarse)
        
        correctorsList = self.fsi.correctorsList
        aPatch = self.fsi.coefficient.aFine

        assert(aPatch.ndim == 1 or aPatch.ndim == 3)
        
        if aPatch.ndim == 1:
            ALocFine = world.ALocFine
        elif aPatch.ndim == 3:
            ALocFine = world.ALocMatrixFine

        localBasis = world.localBasis

        TPrimeCoarsepStartIndices = util.lowerLeftpIndexMap(NPatchCoarse-1, NPatchCoarse)
        TPrimeCoarsepIndexMap = util.lowerLeftpIndexMap(np.ones_like(NPatchCoarse), NPatchCoarse)
        
        TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
        TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)
        
        TPrimeFinepStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)
        TPrimeFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)

        TInd = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, self.iElementPatchCoarse)

        QPatch = np.column_stack(correctorsList)
        
        # This loop can probably be done faster than this. If a bottle-neck, fix!
        Kmsij = np.zeros((NpPatchCoarse, 2**d))
        LTPrimeij = np.zeros((NTPrime, 2**d, 2**d))
        for (TPrimeInd, 
             TPrimeCoarsepStartIndex, 
             TPrimeFinetStartIndex, 
             TPrimeFinepStartIndex) \
             in zip(np.arange(NTPrime),
                    TPrimeCoarsepStartIndices,
                    TPrimeFinetStartIndices,
                    TPrimeFinepStartIndices):

            aTPrime = aPatch[TPrimeFinetStartIndex + TPrimeFinetIndexMap]
            KTPrime = fem.assemblePatchMatrix(NCoarseElement, ALocFine, aTPrime)
            P = localBasis
            Q = QPatch[TPrimeFinepStartIndex + TPrimeFinepIndexMap,:]
            BTPrimeij = np.dot(P.T, KTPrime*Q)
            CTPrimeij = np.dot(Q.T, KTPrime*Q)
            sigma = TPrimeCoarsepStartIndex + TPrimeCoarsepIndexMap
            if TPrimeInd == TInd:
                Kij = np.dot(P.T, KTPrime*P)
                LTPrimeij[TPrimeInd] = CTPrimeij \
                                       - BTPrimeij \
                                       - BTPrimeij.T \
                                       + Kij
                Kmsij[sigma,:] += Kij - BTPrimeij
                
            else:
                LTPrimeij[TPrimeInd] = CTPrimeij
                Kmsij[sigma,:] += -BTPrimeij

        muTPrime = np.zeros(NTPrime)
        for TPrimeInd in np.arange(NTPrime):
            # Solve eigenvalue problem LTPrimeij x = mu_TPrime Mij x
            eigenvalues = scipy.linalg.eigvals(LTPrimeij[TPrimeInd][:-1,:-1], Kij[:-1,:-1])
            muTPrime[TPrimeInd] = np.max(np.real(eigenvalues))

        self.csi = CoarseScaleInformation(Kij, Kmsij, muTPrime)
        
    def computeErrorIndicatorCoarse(self, delta):
        assert(self.csi is not None)

        world = self.world
        NPatchCoarse = world.NWorldCoarse
        NCoarseElement = world.NCoarseElement

        # for localize   #NAMECHANGING horribly #k included here
        iElementPatchCoarse = self.iElementPatchCoarse
        NSubPatchCoarse = self.NPatchCoarse
        iSubPatchCoarse = self.iPatchWorldCoarse

        # gibt den index in coarseTIndexMap an, der fuer das element steht
        elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, iElementPatchCoarse)

        # gibt die gestalt des patches an
        coarseTIndexMap = util.lowerLeftpIndexMap(NSubPatchCoarse - 1, NPatchCoarse - 1)
        # gibt den startindex fuer die obenliegende Gestalt an
        coarseTStartIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse - 1, iSubPatchCoarse)

        muTPrime = self.csi.muTPrime

        numberOfElements = np.shape(coarseTIndexMap)[0]
        NPatchFine = NPatchCoarse * NCoarseElement

        ################## delta ##################
        deltaMax = np.zeros(numberOfElements)
        for i in range(0, numberOfElements):
            # coarse Element Index
            LeftCornerFineElement = util.convertpLinearIndexToCoordIndex(world.NWorldCoarse - 1,
                                                                   coarseTIndexMap[i] + coarseTStartIndex)
            # translate in fine
            LeftCornerFineElement *= np.min(
                NCoarseElement)  # minimum bei eventueller coarseelement1 ungl coraseelement2
            Patching = np.array(
                [np.min(NCoarseElement), np.min(NCoarseElement)])  # minimum bei eventueller ungenauigkeiten
            indexing = util.lowerLeftpIndexMap(Patching - 1, NPatchFine - 1)
            startIndex = util.convertpCoordIndexToLinearIndex(NPatchFine - 1, LeftCornerFineElement)

            deltaMax[i] = np.max(delta[indexing + startIndex])

        ############## ceta #########
        Element = elementCoarseIndex + coarseTStartIndex
        LeftCornerFineElement = util.convertpLinearIndexToCoordIndex(world.NWorldCoarse - 1, Element)
        LeftCornerFineElement *= np.min(NCoarseElement)  # minimum bei eventueller coarseelement1 ungl coraseelement2
        Patching = np.array([np.min(NCoarseElement), np.min(NCoarseElement)])  # minimum bei eventueller ungenauigkeiten
        indexing = util.lowerLeftpIndexMap(Patching - 1, NPatchFine - 1)
        startIndex = util.convertpCoordIndexToLinearIndex(NPatchFine - 1, LeftCornerFineElement)

        ########## total ###########
        epsilonTSquare = np.sum((deltaMax ** 2) * muTPrime)

        return np.sqrt(epsilonTSquare)

