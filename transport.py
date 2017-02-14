import numpy as np
import scipy.sparse as sparse

from world import World
import util
import fem
import linalg

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
            velocityFace = np.sum(AFace*uElement)
            velocityTF[TInd,F] = velocityFace

    return velocityTF

def computeAverageFaceVelocity(NWorldCoarse, velocityTF):
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
