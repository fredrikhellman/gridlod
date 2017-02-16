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

def faceElementIndices(NPatchCoarse, NCoarseElement, k, boundary):
    '''Return indices of fine elements adjacent to face (k, boundary) for all coarse triangles

    Return type shape: (NtCoarse, Number of elements of face)
    '''
    NPatchFine = NPatchCoarse*NCoarseElement

    NFace = np.array(NCoarseElement)
    NFace[k] = 1
    
    tPatchBasis = util.linearpIndexBasis(NPatchFine-1)
    tStepToFace = boundary*tPatchBasis[k]*(NCoarseElement[k]-1)

    tIndexMap = tStepToFace + util.lowerLeftpIndexMap(NFace-1, NPatchFine-1)
    tStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
    tIndices = np.add.outer(tStartIndices, tIndexMap)
    
    return tIndices

def faceElementPointIndices(NPatchCoarse, NCoarseElement, k, boundary):
    '''Return indices of points in fine elements adjacent to face (k, boundary) for all coarse triangles

    Return type shape: (NtCoarse, Number of elements of face, 2**d)
    '''
    NPatchFine = NPatchCoarse*NCoarseElement
    
    tIndices = faceElementIndices(NPatchCoarse, NCoarseElement, k, boundary)
    tpIndexMap = util.lowerLeftpIndexMap(NPatchFine-1, NPatchFine)
    elementpIndices = util.elementpIndexMap(NPatchFine)

    pIndices = np.add.outer(tpIndexMap[tIndices], elementpIndices)
    
    return pIndices

def harmonicMeanOverFaces(NPatchCoarse, NCoarseElement, k, boundary, aPatch):
    NPatchFine = NPatchCoarse*NCoarseElement
    
    TPatchBasis = util.linearpIndexBasis(NPatchCoarse-1)

    NPatchBottom = np.array(NPatchCoarse)
    NPatchBottom[k] = 1
    NtBottom = np.prod(NPatchBottom)

    NPatchBase = np.array(NPatchCoarse)
    NPatchBase[k] -= 1
        
    TIndBase = util.lowerLeftpIndexMap(NPatchBase-1, NPatchCoarse-1)
    TIndBottom = util.lowerLeftpIndexMap(NPatchBottom-1, NPatchCoarse-1)

    t0Faces = faceElementIndices(NPatchCoarse, NCoarseElement, k, 0)
    t1Faces = faceElementIndices(NPatchCoarse, NCoarseElement, k, 1)

    aH0Faces = aPatch[t0Faces[TPatchBasis[k] + TIndBase]]
    aH1Faces = aPatch[t1Faces[TIndBase]]

    if boundary==0:
        abFaces = aPatch[t0Faces]
        abFaces[TPatchBasis[k] + TIndBase] = 2*aH0Faces*aH1Faces/(aH0Faces + aH1Faces)
    elif boundary==1:
        abFaces = aPatch[t1Faces]
        abFaces[TIndBase] = 2*aH0Faces*aH1Faces/(aH0Faces + aH1Faces)

    return abFaces

def computeHarmonicMeanFaceVelocity(NWorldCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch):
    NWorldFine = NWorldCoarse*NCoarseElement
    
    NtCoarse = np.prod(NPatchCoarse)
    d = np.size(NWorldCoarse)
    
    velocityTF = np.zeros([NtCoarse, 2*d])
    
    CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NWorldFine)
    for k in range(d):
        for boundary in [0, 1]:
            boundaryMap = np.zeros([d, 2], dtype='bool')
            boundaryMap[k] = [boundary==0, boundary==1]
            B = fem.assemblePatchBoundaryMatrix(np.ones_like(NPatchCoarse), CLocGetter, None, boundaryMap).todense()
        
            aFaces = harmonicMeanOverFaces(NPatchCoarse, NCoarseElement, k, boundary, aPatch)

            pIndices = faceElementPointIndices(NPatchCoarse, NCoarseElement, k, boundary)
            uFaces = uPatch[pIndices]

            velocityTF[:,2*k+boundary] = -np.einsum('Tfp, kp, Tf -> T', uFaces, B, aFaces)

    return velocityTF
    
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
