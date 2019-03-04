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

def ritzProjectionToFinePatch(patch,
                              APatchFull,
                              bPatchFullList,
                              IPatch,
                              saddleSolver=None):
    if saddleSolver is None:
        saddleSolver = SchurComplementSolver()  # Fast for small patch problems

    world = patch.world
    d = np.size(patch.NPatchCoarse)
    NPatchFine = patch.NPatchFine
    NpFine = patch.NpFine

    # Find what patch faces are common to the world faces, and inherit
    # boundary conditions from the world for those. For the other
    # faces, all DoFs fixed (Dirichlet)
    boundaryMapWorld = world.boundaryConditions==0

    inherit0 = patch.iPatchWorldCoarse==0
    inherit1 = (patch.iPatchWorldCoarse + patch.NPatchCoarse)==world.NWorldCoarse

    boundaryMap = np.ones([d, 2], dtype='bool')
    boundaryMap[inherit0,0] = boundaryMapWorld[inherit0,0]
    boundaryMap[inherit1,1] = boundaryMapWorld[inherit1,1]

    # Using schur complement solver for the case when there are no
    # Dirichlet conditions does not work. Fix if necessary.
    assert(np.any(boundaryMap == True))

    fixed = util.boundarypIndexMap(NPatchFine, boundaryMap)

    #projectionsList = saddleSolver.solve(APatch, IPatch, bPatchList)

    projectionsList = saddleSolver.solve(APatchFull, IPatch, bPatchFullList, fixed, patch.NPatchCoarse, world.NCoarseElement)

    return projectionsList

class CoarseScaleInformation:
    def __init__(self, Kij, Kmsij, muTPrime):
        self.Kij = Kij
        self.Kmsij = Kmsij
        self.muTPrime = muTPrime

class CoarseScaleInformationFlux:
    def __init__(self, correctorFluxTF, basisFluxTF):
        self.correctorFluxTF = correctorFluxTF
        self.basisFluxTF = basisFluxTF

def computeElementCorrector(patch, IPatch, aPatch, ARhsList=None, MRhsList=None, saddleSolver=None):
    '''Compute the fine correctors over a patch.

    Compute the correctors

    (A \nabla Q_T_j, \nabla vf)_{U_K(T)} = (A \nabla ARhs_j, \nabla vf)_{T} + (MRhs_j, vf)_{T}
    '''

    while callable(IPatch):
        IPatch = IPatch()

    while callable(aPatch):
        aPatch = aPatch()


    assert(ARhsList is not None or MRhsList is not None)
    numRhs = None

    if ARhsList is not None:
        assert(numRhs is None or numRhs == len(ARhsList))
        numRhs = len(ARhsList)

    if MRhsList is not None:
        assert(numRhs is None or numRhs == len(MRhsList))
        numRhs = len(MRhsList)

    world = patch.world
    NCoarseElement = world.NCoarseElement
    NPatchCoarse = patch.NPatchCoarse
    d = np.size(NCoarseElement)

    NPatchFine = NPatchCoarse*NCoarseElement
    NtFine = np.prod(NPatchFine)
    NpFineCoarseElement = np.prod(NCoarseElement+1)
    NpCoarse = np.prod(NPatchCoarse+1)
    NpFine = np.prod(NPatchFine+1)

    assert(aPatch.shape[0] == NtFine)
    assert(aPatch.ndim == 1 or aPatch.ndim == 3)

    if aPatch.ndim == 1:
        ALocFine = world.ALocFine
    elif aPatch.ndim == 3:
        ALocFine = world.ALocMatrixFine

    MLocFine = world.MLocFine

    iElementPatchCoarse = patch.iElementPatchCoarse
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

    correctorsList = ritzProjectionToFinePatch(patch,
                                               APatchFull,
                                               bPatchFullList,
                                               IPatch,
                                               saddleSolver)

    return correctorsList

def computeBasisCorrectors(patch, IPatch, aPatch, saddleSolver=None):
    '''Compute the fine basis correctors over the patch.

    Compute the correctors Q_T\lambda_i:

    (A \nabla Q_T lambda_j, \nabla vf)_{U_K(T)} = (A \nabla lambda_j, \nabla vf)_{T}
    '''

    ARhsList = list(patch.world.localBasis.T)

    return computeElementCorrector(patch, IPatch, aPatch, ARhsList, saddleSolver=None)

def computeErrorIndicatorFine(patch, lambdasList, correctorsList, aPatchOld, aPatchNew):
    ''' Compute the fine error idicator e(T) for a general Ritz-projected 
    function lambda.

    This requires lambdas, its correctors and the new and old coefficient.
    '''

    while callable(aPatchOld):
        aPatchOld = aPatchOld()

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    NPatchCoarse = patch.NPatchCoarse
    world = patch.world
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement

    a = aPatchNew

    assert(a.ndim == 1 or a.ndim == 3)

    if a.ndim == 1:
        ALocFine = world.ALocFine
    else:
        ALocFine = world.ALocMatrixFine
        
    P = np.column_stack(lambdasList)

    aTilde = aPatchOld

    TFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)
    iElementPatchFine = patch.iElementPatchCoarse*NCoarseElement
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

    Q = np.column_stack(correctorsList)
    QT = Q[TFinepStartIndex + TFinepIndexMap,:]

    A = np.dot((P-QT).T, TNorm*(P-QT)) + np.dot(Q.T, PatchNorm*Q) - np.dot(QT.T, TNorm*QT)
    B = np.dot(P.T, BNorm*P)

    eigenvalues = scipy.linalg.eigvals(A[:-1,:-1], B[:-1,:-1])
    epsilonTSquare = np.max(np.real(eigenvalues))

    return np.sqrt(epsilonTSquare)

def computeBasisErrorIndicatorFine(patch, correctorsList, aPatchOld, aPatchNew):
    ''' Compute the fine error idicator e(T) for a corrected basis.
    '''

    lambdasList = list(patch.world.localBasis.T)

    return computeErrorIndicatorFine(patch, lambdasList, correctorsList, aPatchOld, aPatchNew)

def computeErrorIndicatorCoarseFromGreeks(patch, muTPrime, greeksPatch):
    '''Compute the coarse error idicator E(T) from the "greeks" delta and kappa,
    where

    deltaMaxTPrime = || ANew^{-1/2} (ANew - AOld) AOld^{-1/2} ||_max(TPrime)
                                                     over all coarse elements TPrime in the patch

    kappaMaxT = || AOld^{1/2) ANew^{-1/2} ||_max(T)
                                 over the patch coarse T (patch.iPatchWorldCoarse)

    This requires muTPrime from CSI and the new and old coefficient.

    greeksPatch should either be
      1. callable and returns (kappaMaxT, deltaMaxTPrime)
      2. the tuple (kappaMaxT, deltaMaxTPrime)

    This function is convenient if the greeks can be computed without
    the need of AOld or ANew, for example from the Jacobian in the
    domain mapping case.

    '''

    while callable(greeksPatch):
        greeksPatch = greeksPatch()

    deltaMaxTPrime, kappaMaxT = greeksPatch

    epsilonTSquare = kappaMaxT**2 * np.sum((deltaMaxTPrime**2)*muTPrime)

    return np.sqrt(epsilonTSquare)

def computeErrorIndicatorCoarseFromCoefficients(patch, muTPrime, aPatchOld, aPatchNew):
    ''' Compute the coarse error idicator E(T) with explicit value of AOld and ANew.

    This requires muTPrime from CSI and the new and old coefficient.
    '''

    while callable(muTPrime):
        muTPrime = muTPrime()

    while callable(aPatchOld):
        aPatchOld = aPatchOld()

    while callable(aPatchNew):
        aPatchNew = aPatchNew()

    aOld = aPatchOld
    aNew = aPatchNew

    assert(aNew.ndim == 1) # Matrix-valued A not supported in thus function yet

    world = patch.world
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement
    iElementPatchCoarse = patch.iElementPatchCoarse

    elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, iElementPatchCoarse)

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
    aTPrime = aNew[TPrimeIndices]
    aOldTPrime = aOld[TPrimeIndices]

    deltaMaxTPrime = np.max(np.abs((aTPrime - aOldTPrime)/np.sqrt(aTPrime*aOldTPrime)), axis=1)
    kappaMaxT = np.sqrt(np.max(np.abs(aOldTPrime[elementCoarseIndex]/aTPrime[elementCoarseIndex])))

    return computeErrorIndicatorCoarseFromGreeks(patch, muTPrime, (deltaMaxTPrime, kappaMaxT))

def computeCoarseQuantities(patch, lambdasList, correctorsList, aPatch):
    '''Compute coarse quantities for pairs of lambdas and correctors

    Compute the tensors (T is implcit by the patch definition):

    KTij   = (A \nabla lambda_j, \nabla lambda_i)_{T}
    KmsTij = (A \nabla (lambda_j - Q_T lambda_j), \nabla lambda_i)_{U_k(T)}
    muTT'  = max_{w_H} || A (\chi_T \nabla - \nabla Q_T) w_H) ||^2_T' / || A \nabla w_H ||^2_T

    and store them in the self.csi object. See
    notes/coarse_quantities*.pdf for a description.

    Auxiliary quantities are computed, but not saved, e.g.

    LTT'ij = (A \nabla (chi_T - Q_T)lambda_j, \nabla (chi_T - Q_T) lambda_j)_{T'}
    '''

    # This function could probably also be generalized and made work
    # on general functions \lambda instead of specifically the basis
    # functions.
    
    while callable(aPatch):
        aPatch = aPatch()

    world = patch.world
    NCoarseElement = world.NCoarseElement
    NPatchCoarse = patch.NPatchCoarse
    NPatchFine = NPatchCoarse*NCoarseElement

    NTPrime = np.prod(NPatchCoarse)
    NpPatchCoarse = np.prod(NPatchCoarse+1)

    d = np.size(NPatchCoarse)

    assert(aPatch.ndim == 1 or aPatch.ndim == 3)

    if aPatch.ndim == 1:
        ALocFine = world.ALocFine
    elif aPatch.ndim == 3:
        ALocFine = world.ALocMatrixFine

    lambdas = np.column_stack(lambdasList)
    numLambdas = len(lambdasList)
    
    
    TPrimeCoarsepStartIndices = util.lowerLeftpIndexMap(NPatchCoarse-1, NPatchCoarse)
    TPrimeCoarsepIndexMap = util.lowerLeftpIndexMap(np.ones_like(NPatchCoarse), NPatchCoarse)

    TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

    TPrimeFinepStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)
    TPrimeFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)

    TInd = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, patch.iElementPatchCoarse)

    QPatch = np.column_stack(correctorsList)

    # This loop can probably be done faster than this. If a bottle-neck, fix!
    Kmsij = np.zeros((NpPatchCoarse, numLambdas))
    LTPrimeij = np.zeros((NTPrime, numLambdas, numLambdas))
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
        P = lambdas
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

            # Fix this some other way....
            # Kmsij is special for the basis-case
            if numLambdas == 2**d:
                Kmsij[sigma,:] += Kij - BTPrimeij

        else:
            LTPrimeij[TPrimeInd] = CTPrimeij
            
            if numLambdas == 2**d:
                Kmsij[sigma,:] += -BTPrimeij

    muTPrime = np.zeros(NTPrime)
    for TPrimeInd in np.arange(NTPrime):
        cutRows = 0
        while np.linalg.cond(Kij[cutRows:,cutRows:]) > 1e16:
            cutRows = cutRows + 1
            
        # Solve eigenvalue problem LTPrimeij x = mu_TPrime Mij x
        eigenvalues = scipy.linalg.eigvals(LTPrimeij[TPrimeInd][cutRows:,cutRows:], Kij[cutRows:,cutRows:])
        muTPrime[TPrimeInd] = np.max(np.real(eigenvalues))

    return CoarseScaleInformation(Kij, Kmsij, muTPrime)

def computeBasisCoarseQuantities(patch, correctorsList, aPatch):
    ''' Compute the coarse quantities for the local basis and its correctors
    '''

    lambdasList = list(patch.world.localBasis.T)

    return computeCoarseQuantities(patch, lambdasList, correctorsList, aPatch)


def computeCoarseQuantitiesFlux(patch, correctorsList, aPatch):

    while callable(aPatch):
        aPatch = aPatch()

    world = patch.world
    NCoarseElement = world.NCoarseElement
    NPatchCoarse = patch.NPatchCoarse
    NPatchFine = NPatchCoarse*NCoarseElement

    correctorsList = correctorsList
    QPatch = np.column_stack(correctorsList)

    localBasis = world.localBasis

    TPrimeFinepStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)
    TPrimeFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)

    TInd = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, patch.iElementPatchCoarse)

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
    return CoarseScaleInformationFlux(correctorFluxTF, basisFluxTF)
