import numpy as np
import scipy.sparse as sparse

from world import World
import util
import fem
import linalg

def computeElementFaceVelocityFromSigma(NWorldCoarse, sigmaFluxT):
    d = np.size(NWorldCoarse)

    NtCoarse = np.prod(NWorldCoarse)
    velocityTF = np.zeros([NtCoarse, 2*d])

    pBasis = util.linearpIndexBasis(np.ones_like(NWorldCoarse))
    for k in range(d):
        N = np.ones_like(NWorldCoarse)
        N[k] = 0
        bottomp = util.lowerLeftpIndexMap(N, np.ones_like(N))
        topp = bottomp + pBasis[k]

        NFace = np.array(NWorldCoarse)
        NFace[k] = 1
        faceArea = np.prod(1./NFace)
        
        velocityTF[:,2*k] = faceArea*np.mean(-sigmaFluxT[:,bottomp], axis=1)
        velocityTF[:,2*k + 1] = faceArea*np.mean(-sigmaFluxT[:,topp], axis=1)

    return velocityTF

def computeHarmonicMeanAverageFaceVelocity(NWorldCoarse, NCoarseElement, aWorld, uWorld):

    NWorldFine = NWorldCoarse*NCoarseElement

    d = np.size(NWorldFine)
    
    pWorldBasis = util.linearpIndexBasis(NWorldFine)
    tWorldBasis = util.linearpIndexBasis(NWorldFine-1)

    CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NWorldFine)
    
    for k in range(d):
        # Interior
        NFace = np.array(NCoarseElement)
        NFace[k] = 1

        tStepToFacei = tWorldBasis[k]*(NCoarseElement[k]-1)
        tStepToFacej = 0

        pStepToFacei = pWorldBasis[k]*(NCoarseElement[k]-1)
        pStepToFacej = 0
        
        boundaryMap = np.zeros([d, 2], dtype='bool')

        boundaryMap[k] = [False, True]
        Bi = fem.assemblePatchBoundaryMatrix(np.ones_like(NWorldCoarse), CLocGetter, None, boundaryMap).todense()
        tiIndexMap = tStepToFacei + util.lowerLeftpIndexMap(NFace-1, NWorldFine-1)
        piIndexMap = pStepToFacei + util.lowerLeftpIndexMap(NFace, NCoarseElement)
        
        boundaryMap[k] = [True, False]
        Bj = fem.assemblePatchBoundaryMatrix(np.ones_like(NWorldCoarse), CLocGetter, None, boundaryMap).todense()
        tjIndexMap = tWorldBasis[k] + tiIndexMap
        pjIndexMap = pWorldBasis[k] + piIndexMap

        tStartIndices = util.pIndexMap(NWorldCoarse-2, NWorldFine-1, NCoarseElement)
        pStartIndices = util.pIndexMap(NWorldCoarse-2, NWorldFine, NCoarseElement)

        tiIndices = np.add.outer(tStartIndices, tiIndexMap)
        tjIndices = np.add.outer(tStartIndices, tjIndexMap)
        
        aHarmonic = 2*(aWorld[tiIndices]*aWorld[tjIndices])/(aWorld[tiIndices] + aWorld[tjIndices])

        faceElementpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NFace), NFace)
        faceElementpStartIndices = util.lowerLeftpIndexMap(NFace-1, NFace)
        faceElementpIndices = np.add.outer(faceElementpStartIndices, faceElementpIndexMap)


        # pi/jfIndices are 3d-tensors: triangle, face, point dof
        def oneSide(pIndexMap, B):
            pIndices = np.add.outer(pStartIndices, pIndexMap)
            pfIndices = pIndices[:,faceElementpIndices]
            upf = uWorld[pfIndices]
            return np.einsum('Tfp, kp, Tf -> T', upf, B, aHarmonic)

        
        velocityFi = oneSide(piIndexMap, Bi)
        velocityFj = oneSide(pjIndexMap, Bj)

        avgVelocityF = velocityFi-velocityFj
        
        ## HOW TO COMPUTE B*u, for each fine face?
    
def computeElementFaceVelocity(NWorldCoarse, NCoarseElement, aWorld, uWorld):
    '''Per element, compute face normal velocity integrals.'''
    NWorldFine = NWorldCoarse*NCoarseElement
    
    NtCoarse = np.prod(NWorldCoarse)
    d = np.size(NWorldCoarse)

    velocityTF = np.zeros([NtCoarse, 2*d])

    TFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NWorldFine)
    TFinepStartIndices = util.pIndexMap(NWorldCoarse-1, NWorldFine, NCoarseElement)

    TFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NWorldFine-1)
    TFinetStartIndices = util.pIndexMap(NWorldCoarse-1, NWorldFine-1, NCoarseElement)

    faces = np.arange(2*d)

    CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NWorldFine)
    boundaryMap = np.zeros([d, 2], dtype='bool')
    for TInd in np.arange(NtCoarse):
        aElement = aWorld[TFinetStartIndices[TInd] + TFinetIndexMap]
        uElement = uWorld[TFinepStartIndices[TInd] + TFinepIndexMap]
        for F in faces:
            boundaryMap[:] = False
            boundaryMap.flat[F] = True
            AFace = fem.assemblePatchBoundaryMatrix(NCoarseElement, CLocGetter, aElement, boundaryMap)
            velocityFace = -np.sum(AFace*uElement) ### Too big probably.....
            velocityTF[TInd,F] = velocityFace

    return velocityTF

def computeAverageFaceVelocity(NWorldCoarse, velocityTF):
    # Note  I: these velocities are not conservative and do not fulfill
    #          strongly with Neumann boundary conditions
    #
    # Note II: the velocities are integrated and hence already scaled with face
    #          area
    d = np.size(NWorldCoarse)
    b = util.linearpIndexBasis(NWorldCoarse-1)

    avgVelocityTF = np.array(velocityTF)
    for k in range(d):
        NWorldBase = np.array(NWorldCoarse-1)
        NWorldBase[k] -= 1
        TIndBase = util.lowerLeftpIndexMap(NWorldBase, NWorldCoarse-1)
        avg = 0.5*(velocityTF[TIndBase, 2*k + 1] - velocityTF[TIndBase + b[k], 2*k])
        avgVelocityTF[TIndBase, 2*k + 1] = avg
        avgVelocityTF[TIndBase + b[k], 2*k] = -avg

    return avgVelocityTF
        
def computeUpwindSaturation(NWorldCoarse, boundarys, sT, velocityTF):
    d = np.size(NWorldCoarse)
    
    sTF = np.tile(sT[...,None], [1, 2*d])
    basis = util.linearpIndexBasis(NWorldCoarse-1)
    for k in range(d):
        b = basis[k]

        N = np.array(NWorldCoarse-1)
        N[k] -= 1
        TIndBase = util.lowerLeftpIndexMap(N, NWorldCoarse-1)

        N[k] = 0
        TIndBottom = util.lowerLeftpIndexMap(N, NWorldCoarse-1)

        for face in [0,1]:
            faceInd = 2*k+face
            stepToFace = b*(face*(NWorldCoarse[k]-1))
            TIndInterior = (1-face)*b + TIndBase
            TIndBoundary = stepToFace + TIndBottom

            # Interior
            TIndDst = TIndInterior[velocityTF[TIndInterior, faceInd] < 0]
            TIndSrc = TIndDst + (2*face - 1)*b
            sTF[TIndDst, faceInd] = sT[TIndSrc]

            # Boundary
            TIndDst = TIndBoundary[velocityTF[TIndBoundary, faceInd] < 0]
            sTF[TIndDst, faceInd] = boundarys[k, face]
            
    return sTF

def computeElementNetFlux(world, avgVelocityTF, sT, boundarys, fractionalFlow, fCoarse=None, sWellCoarse=None):
    NWorldCoarse = world.NWorldCoarse
    
    fsT = fractionalFlow(sT)
    boundaryfs = fractionalFlow(boundarys)
    fsTF = computeUpwindSaturation(NWorldCoarse, boundaryfs, fsT, avgVelocityTF)

    # Ignore fCoarse, sWellCoarse for now
    assert(fCoarse is None and sWellCoarse is None)

    netFluxT = -np.einsum('tf, tf -> t', fsTF, avgVelocityTF)

    return netFluxT
