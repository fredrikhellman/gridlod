import numpy as np
import scipy.sparse as sparse
import itertools as it

from world import World
import util
import fem
import linalg

def computeBoundaryFlux(world, ROmega):
    NWorldCoarse = world.NWorldCoarse
    NpCoarse = np.prod(NWorldCoarse+1)
    
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
    sigmaFluxT = np.zeros([2**d, NTInds])
    nodeFluxT = np.zeros([2**d, NTInds])

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
        boundary01 = np.zeros(2*d)
        boundary01[:d] = iWorldCoarse==0
        boundary01[d:] = (iWorldCoarse + 1)==NWorldCoarse

        MFluxNodes, MFluxNodesInv, localFluxBoundaryNodes = precomputed[boundaryHash(boundary01)]
        
        RFull = RT[:,TInd]
        RFluxNodes = RFull[localFluxBoundaryNodes]

        # Solve MFluxNodes*sigma = RFluxNodes - MDirichletNodes*sigmaDirichletLocalized
        bFluxNodes = RFluxNodes
        
        #sigmaFlux = np.linalg.solve(MFluxNodes, bFluxNodes)
        sigmaFlux = np.array(np.dot(MFluxNodesInv, bFluxNodes)).T.squeeze()
        sigmaFluxT[localFluxBoundaryNodes, TInd] = sigmaFlux
        nodeFlux = np.dot(MFluxNodes, sigmaFlux)
        nodeFluxT[localFluxBoundaryNodes, TInd] = nodeFlux
        
    return sigmaFluxT, nodeFluxT
