import numpy as np
import scipy.sparse as sparse

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

    class Cache:
        MFluxNodes = None
        localFluxBoundaryNodes = None
        boundary0 = None
        boundary1 = None
        
    def computeStuff(boundary0, boundary1):
        if Cache.MFluxNodes is not None and np.all(Cache.boundary0 == boundary0) and np.all(Cache.boundary1 == boundary1):
            pass
        else:
            localBoundaryMap = np.column_stack([boundary0, boundary1])
            localBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                        localBoundaryMap)
            localFluxBoundaryMap = np.logical_not(np.logical_and(localBoundaryMap, world.boundaryConditions==1))

            localFluxBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                            localFluxBoundaryMap)

            MFluxFull = fem.assemblePatchBoundaryMatrix(np.ones_like(NWorldCoarse),
                                                        MLocGetter,
                                                        boundaryMap=localFluxBoundaryMap).todense()
            MFluxNodes = MFluxFull[localFluxBoundaryNodes][:,localFluxBoundaryNodes]
            Cache.MFluxNodes = MFluxNodes
            Cache.localFluxBoundaryNodes = localFluxBoundaryNodes
            Cache.boundary0 = np.array(boundary0)
            Cache.boundary1 = np.array(boundary1)
            
        return Cache.MFluxNodes, Cache.localFluxBoundaryNodes

    for TInd in TInds:
        print ',',
        iWorldCoarse = util.convertpIndexToCoordinate(NWorldCoarse-1, TInd)
        boundary0 = iWorldCoarse==0
        boundary1 = (iWorldCoarse + 1) == NWorldCoarse

        MFluxNodes, localFluxBoundaryNodes = computeStuff(boundary0, boundary1)
        
        RFull = RT[:,TInd]
        RFluxNodes = RFull[localFluxBoundaryNodes]

        # Solve MFluxNodes*sigma = RFluxNodes - MDirichletNodes*sigmaDirichletLocalized
        bFluxNodes = RFluxNodes
        
        sigmaFlux = np.linalg.solve(MFluxNodes, bFluxNodes)
        sigmaFluxT[localFluxBoundaryNodes, TInd] = sigmaFlux
        nodeFlux = np.dot(MFluxNodes, sigmaFlux)
        nodeFluxT[localFluxBoundaryNodes, TInd] = nodeFlux
        
    return sigmaFluxT, nodeFluxT
