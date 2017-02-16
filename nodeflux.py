import numpy as np
import scipy.sparse as sparse
import itertools as it
import scipy.signal as signal

from world import World
import util
import fem
import linalg


def computeBoundaryFlux(world, ROmega):
    NWorldCoarse = world.NWorldCoarse
    NpCoarse = np.prod(NWorldCoarse+1)

    # Only the dirichlet nodes of ROmega are used....
    assert(np.size(ROmega) == NpCoarse)
    
    boundaryMap = world.boundaryConditions==0
    dirichletNodes = util.boundarypIndexMap(NWorldCoarse, boundaryMap)

    MLocGetter = fem.localBoundaryMassMatrixGetter(NWorldCoarse)
    MDirichletFull = fem.assemblePatchBoundaryMatrix(NWorldCoarse,
                                                     MLocGetter,
                                                     boundaryMap=boundaryMap)

    # Solve MDirichletNodes * sigmaDirichletNodes = RDirichletNodes
    MDirichletNodes = MDirichletFull[dirichletNodes][:,dirichletNodes]
    RDiricheletNodes = ROmega[dirichletNodes]

    sigmaDirichletNodes = sparse.linalg.spsolve(MDirichletNodes, RDiricheletNodes)
    sigmaDirichlet = np.zeros(NpCoarse)
    sigmaDirichlet[dirichletNodes] = sigmaDirichletNodes
    
    return sigmaDirichlet

def computeCoarseElementFlux(world, RT, TInds=None):
    NWorldCoarse = world.NWorldCoarse
    NtCoarse = np.prod(NWorldCoarse)
    d = np.size(NWorldCoarse)

    if TInds is None:
        TInds = np.arange(NtCoarse)

    if RT.ndim == 1:
        RT = RT[:,np.newaxis]
        
    assert(RT.shape[1] == np.size(TInds))
    
    worldDirichletBoundaryMap = world.boundaryConditions==0
    MLocGetter = fem.localBoundaryMassMatrixGetter(NWorldCoarse)

    NTInds = np.size(TInds)
    sigmaFluxT = np.zeros([NTInds, 2**d])
    nodeFluxT = np.zeros([NTInds, 2**d])

    def boundaryHash(boundary01):
        n = np.size(boundary01)
        val = np.dot(boundary01, 2**np.arange(0,n))
        return val

    def computeStuff(boundary01):
        localBoundaryMap = np.reshape(boundary01, [d, 2])
        localBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                    localBoundaryMap)
        localFluxBoundaryMap = np.logical_not(np.logical_and(localBoundaryMap, world.boundaryConditions==1))
        
        localFluxBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                        localFluxBoundaryMap)
        
        MFluxFull = fem.assemblePatchBoundaryMatrix(np.ones_like(NWorldCoarse),
                                                    MLocGetter,
                                                    boundaryMap=localFluxBoundaryMap).todense()
        MFluxNodes = MFluxFull[localFluxBoundaryNodes][:,localFluxBoundaryNodes]
        MFluxNodesInv = np.linalg.inv(MFluxNodes)
            
        return MFluxNodes, MFluxNodesInv, localFluxBoundaryNodes

    precomputed = {}
    for boundary01 in it.product(*([[0,1]]*(2*d))):
        boundary01 = np.array(boundary01)
        precomputed[boundaryHash(boundary01)] = computeStuff(boundary01)
        
    iWorldCoarses = util.convertpIndexToCoordinate(NWorldCoarse-1, TInds)
    for TInd in TInds:
        iWorldCoarse = iWorldCoarses[:,TInd]

        boundary01 = np.zeros(2*d, dtype='int64')
        boundary01[::2] = iWorldCoarse==0
        boundary01[1::2] = (iWorldCoarse + 1)==NWorldCoarse

        MFluxNodes, MFluxNodesInv, localFluxBoundaryNodes = precomputed[boundaryHash(boundary01)]
        
        RFull = RT[:,TInd]
        RFluxNodes = RFull[localFluxBoundaryNodes]

        # Solve MFluxNodes*sigma = RFluxNodes - MDirichletNodes*sigmaDirichletLocalized
        bFluxNodes = RFluxNodes
        
        #sigmaFlux = np.linalg.solve(MFluxNodes, bFluxNodes)
        sigmaFlux = np.array(np.dot(MFluxNodesInv, bFluxNodes)).T.squeeze()
        sigmaFluxT[TInd, localFluxBoundaryNodes] = sigmaFlux
        nodeFlux = np.dot(MFluxNodes, sigmaFlux)
        nodeFluxT[TInd, localFluxBoundaryNodes] = nodeFlux
        
    return sigmaFluxT, nodeFluxT

def computeMeanElementwiseQuantity(world, boundaryFsValues, fsT):
    NWorldCoarse = world.NWorldCoarse
    d = np.size(NWorldCoarse)
    
    boundaryConditions = world.boundaryConditions
    
    fsCube = fsT.reshape(NWorldCoarse, order='F')
    fsBorderCube = np.zeros(NWorldCoarse+2)
    fsBorderCube[[slice(1,-1)]*d] = fsCube

    # Set Dirichlet boundary conditions
    for k in range(d):
        index = [slice(None)]*d
        if boundaryConditions[k,0]==0:
            index[k] = 0
            fsBorderCube[index] = boundaryFsValues[k,0]
        if boundaryConditions[k,1]==0:
            index[k] = -1
            fsBorderCube[index] = boundaryFsValues[k,1]

    # For other boundary segments, copy
    for k in range(d):
        index = [slice(None)]*d
        indexInner = [slice(None)]*d
        if boundaryConditions[k,0] != 0:
            index[k] = 0
            indexInner[k] = 1
            fsBorderCube[index] = fsBorderCube[indexInner]
        if boundaryConditions[k,1] != 0:
            index[k] = -1
            indexInner[k] = -2
            fsBorderCube[index] = fsBorderCube[indexInner]
            
    # Apply convolution
    avgFilter = np.ones([2]*d)/(2**d)
    nodeFsCube = signal.convolve(fsBorderCube, avgFilter, mode='valid')
    nodeFs = nodeFsCube.flatten(order='F')
    return nodeFs

def computeElementNetFlux(world, boundaryFsValues, fsT, nodeFluxT):
    NWorldCoarse = world.NWorldCoarse
    NtCoarse = np.prod(NWorldCoarse)

    nodeFs = computeMeanElementwiseQuantity(world, boundaryFsValues, fsT)

    TpIndexMap = util.elementpIndexMap(NWorldCoarse)
    TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)
    TpIndices = np.add.outer(TpStartIndices, TpIndexMap)

    nodeFsT = nodeFs[TpIndices]

    netFluxT = np.einsum('ij, ij -> i', nodeFsT, nodeFluxT)

    return netFluxT
