import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg
import gc
import warnings

import fem
import util
import linalg
import coef
import transport

# Saddle point problem solvers
class nullspaceSolver:
    def __init__(self, NPatchCoarse, NCoarseElement):
        NPatchFine = NPatchCoarse*NCoarseElement
        self.coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)

    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        raise(NotImplementedError('Not maintained'))
        return linalg.saddleNullSpace(A, I, bList, self.coarseNodes)

class nullspaceOneLevelHierarchySolver:
    def __init__(self, NPatchCoarse, NCoarseElement):
        NPatchFine = NPatchCoarse*NCoarseElement
        self.coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)
        self.PPatch = fem.assembleProlongationMatrix(NPatchCoarse, NCoarseElement)
        # It is time consuming to create PPatch. Perhaps another
        # solver of this kind can be implemented, but where PPatch is
        # implicitly defined instead...
        
    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        return linalg.saddleNullSpaceHierarchicalBasis(A, I, self.PPatch, bList, self.coarseNodes, fixed)

class nullspaceSeveralLevelsHierarchySolver:
    def __init__(self, NPatchCoarse, NCoarseElement):
        NPatchFine = NPatchCoarse*NCoarseElement
        self.coarseNodes = util.fillpIndexMap(NPatchCoarse, NPatchFine)
        self.PHier = fem.assembleHierarchicalBasisMatrix(NPatchCoarse, NCoarseElement)

    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        raise(NotImplementedError('Not maintained'))
        return linalg.saddleNullSpaceGeneralBasis(A, I, self.PHier, bList, self.coarseNodes)

class blockDiagonalPreconditionerSolver:
    def __init__(self):
        pass
    
    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        raise(NotImplementedError('Fix this!'))
        return linalg.solveWithBlockDiagonalPreconditioner(A, I, bList)

class schurComplementSolver:
    def __init__(self, NCache=None):
        if NCache is not None:
            self.cholCache = linalg.choleskyCache(NCache)
        else:
            self.cholCache = None
    
    def solve(self, A, I, bList, fixed, NPatchCoarse=None, NCoarseElement=None):
        return linalg.schurComplementSolve(A, I, bList, fixed, NPatchCoarse, NCoarseElement, self.cholCache)

class directSolver:
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

    #saddleSolver = nullspaceOneLevelHierarchySolver(NPatchCoarse, NCoarseElement)
    saddleSolver = schurComplementSolver()  # Fast for small patch problems
    
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
    def __init__(self, coefficientPatch, correctorsList):
        self.coefficient = coefficientPatch
        self.correctorsList = correctorsList

class CoarseScaleInformation:
    def __init__(self, Kij, Kmsij, muTPrime, correctorFluxTF, basisFluxTF, rCoarse=None):
        self.Kij = Kij
        self.Kmsij = Kmsij
        #self.LTPrimeij = LTPrimeij
        self.muTPrime = muTPrime
        self.rCoarse = rCoarse
        self.correctorFluxTF = correctorFluxTF
        self.basisFluxTF = basisFluxTF


        
class elementCorrector:
    def __init__(self, world, k, iElementWorldCoarse, saddleSolver=None):
        self.k = k
        self.iElementWorldCoarse = iElementWorldCoarse[:]
        self.world = world

        # Compute (NPatchCoarse, iElementPatchCoarse) from (k, iElementWorldCoarse, NWorldCoarse)
        d = np.size(iElementWorldCoarse)
        NWorldCoarse = world.NWorldCoarse
        iPatchWorldCoarse = np.maximum(0, iElementWorldCoarse - k).astype('int64')
        iEndPatchWorldCoarse = np.minimum(NWorldCoarse - 1, iElementWorldCoarse + k).astype('int64') + 1
        self.NPatchCoarse = iEndPatchWorldCoarse-iPatchWorldCoarse
        self.iElementPatchCoarse = iElementWorldCoarse - iPatchWorldCoarse
        self.iPatchWorldCoarse = iPatchWorldCoarse

        if saddleSolver == None:
            self._saddleSolver = schurComplementSolver()
        else:
            self._saddleSolver = saddleSolver
            
    @property
    def saddleSolver(self):
        return self._saddleSolver

    @saddleSolver.setter
    def saddleSolver(self, value):
        self._saddleSolver = value
            
    def computeElementCorrector(self, coefficientPatch, IPatch, ARhsList=None, MRhsList=None):
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
        return correctorsList

    def computeCorrectors(self, coefficientPatch, IPatch):
        '''Compute the fine correctors over the patch.

        Compute the correctors Q_T\lambda_i (T is given by the class instance):

        (A \nabla Q_T lambda_j, \nabla vf)_{U_K(T)} = (A \nabla lambda_j, \nabla vf)_{T}

        and store them in the self.fsi object, together with the extracted A|_{U_k(T)}
        '''
        d = np.size(self.NPatchCoarse)
        ARhsList = map(np.squeeze, np.hsplit(self.world.localBasis, 2**d))

        correctorsList = self.computeElementCorrector(coefficientPatch, IPatch, ARhsList)
        
        self.fsi = FineScaleInformation(coefficientPatch, correctorsList)
        
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
        assert(hasattr(self, 'fsi'))

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

        if aPatch.ndim == 1:
            # Face flux computations are only possible for scalar coefficients
            # Compute coarse element face flux for basis and Q (not yet implemented for R)
            correctorFluxTF = transport.computeHarmonicMeanFaceFlux(world.NWorldCoarse,
                                                                    NPatchCoarse,
                                                                    NCoarseElement, aPatch, QPatch)

            # Need to compute over at least one fine element over the
            # boundary of the main element to make the harmonic average
            # right.  Note: We don't do this for correctorFluxTF, beause
            # we do not have access to a outside the patch...
            localBasisExtended = np.zeros_like(QPatch)
            localBasisExtended[TPrimeFinepStartIndices[TInd] + TPrimeFinepIndexMap,:] = localBasis
            basisFluxTF = transport.computeHarmonicMeanFaceFlux(world.NWorldCoarse,
                                                                NPatchCoarse,
                                                                NCoarseElement, aPatch, localBasisExtended)[:,TInd,:]
        else:
            # For non-scalar A we cannot compute these flux-quantities (yet)
            basisFluxTF = None
            correctorFluxTF = None

        if isinstance(self.fsi.coefficient, coef.coefficientCoarseFactorAbstract):
            rCoarse = self.fsi.coefficient.rCoarse
        else:
            rCoarse = None
        self.csi = CoarseScaleInformation(Kij, Kmsij, muTPrime, correctorFluxTF, basisFluxTF, rCoarse)

    def clearFineQuantities(self):
        assert(hasattr(self, 'fsi'))
        del self.fsi

    def computeErrorIndicatorFineWithLagging(self, a, aTilde):
        assert(hasattr(self, 'csi'))

        assert(a.ndim == 1) # Matrix-valued A not supported in thus function yet
        
        world = self.world
        NPatchCoarse = self.NPatchCoarse
        NCoarseElement = world.NCoarseElement
        NPatchFine = NPatchCoarse*NCoarseElement
        iElementPatchCoarse = self.iElementPatchCoarse

        elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, iElementPatchCoarse)
        
        TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
        TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

        muTPrime = self.csi.muTPrime

        TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
        aTPrime = a[TPrimeIndices]
        aTildeTPrime = aTilde[TPrimeIndices]
        
        deltaMaxNormTPrime = np.max(np.abs((aTPrime - aTildeTPrime)/np.sqrt(aTPrime*aTildeTPrime)), axis=1)
        theOtherUnnamedFactorTPrime = np.max(np.abs(aTPrime[elementCoarseIndex]/aTildeTPrime[elementCoarseIndex]))

        epsilonTSquare = theOtherUnnamedFactorTPrime * \
                         np.sum((deltaMaxNormTPrime**2)*muTPrime)

        return np.sqrt(epsilonTSquare)
        
    def computeErrorIndicator(self, rCoarseNew):
        assert(hasattr(self, 'csi'))
        assert(self.csi.rCoarse is not None)
        
        world = self.world
        NPatchCoarse = self.NPatchCoarse
        NCoarseElement = world.NCoarseElement
        iElementPatchCoarse = self.iElementPatchCoarse

        elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, iElementPatchCoarse)
        
        rCoarse = self.csi.rCoarse
        muTPrime = self.csi.muTPrime
        deltaMaxNormTPrime = np.abs((rCoarseNew - rCoarse)/np.sqrt(rCoarseNew*rCoarse))
        
        epsilonTSquare = rCoarseNew[elementCoarseIndex]/rCoarse[elementCoarseIndex] * \
                         np.sum((deltaMaxNormTPrime**2)*muTPrime)

        return np.sqrt(epsilonTSquare)

    def computeErrorIndicatorFine(self, coefficientNew):
        assert(hasattr(self, 'fsi'))

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
    
# def computeelementCorrectorDirichletBC(NPatchCoarse,
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
