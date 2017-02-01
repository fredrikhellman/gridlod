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

def computeCoarseElementFlux(world, ROmega, RT, TInds=None):
    NWorldCoarse = world.NWorldCoarse
    NtCoarse = np.prod(NWorldCoarse)
    d = np.size(NWorldCoarse)

    if TInds is None:
        TInds = np.arange(NtCoarse)

    if RT.ndim == 1:
        RT = RT[:,np.newaxis]
        
    assert(RT.shape[1] == np.size(TInds))
    
    # Compute boundary flux
    sigmaDirichlet = computeBoundaryFlux(world, ROmega)
    
    worldDirichletBoundaryMap = world.boundaryConditions==0
    MLocGetter = fem.localBoundaryMassMatrixGetter(NWorldCoarse)

    NTInds = np.size(TInds)
    sigmaFluxT = np.zeros([2**d, NTInds])

    for TInd in TInds:
        print ',',
        iWorldCoarse = util.convertpIndexToCoordinate(NWorldCoarse-1, TInd)
        boundary0 = iWorldCoarse==0
        boundary1 = (iWorldCoarse + 1) == NWorldCoarse

        # Much of this is equal from iteration to iteration, can be sped up if needed
        localBoundaryMap = np.column_stack([boundary0, boundary1])
        localBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                    localBoundaryMap)

        localDirichletBoundaryMap = np.logical_and(localBoundaryMap, worldDirichletBoundaryMap)
        localDirichletBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                             localDirichletBoundaryMap)
        ###
        # Why not do like this? Just don't use boundary flux. Only compute element flux
        localDirichletBoundaryNodes = np.array([], dtype='int64')
        ###
        
        MDirichletFull = fem.assemblePatchBoundaryMatrix(np.ones_like(NWorldCoarse),
                                                         MLocGetter,
                                                         boundaryMap=localDirichletBoundaryMap)

        localFluxBoundaryMap = np.logical_not(localBoundaryMap)
        ###
        localFluxBoundaryMap = np.logical_not(np.logical_and(localBoundaryMap, world.boundaryConditions==1))
        ###
        localFluxBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                        localFluxBoundaryMap)

        MFluxFull = fem.assemblePatchBoundaryMatrix(np.ones_like(NWorldCoarse),
                                                    MLocGetter,
                                                    boundaryMap=localFluxBoundaryMap)
        MFluxNodes = MFluxFull[localFluxBoundaryNodes][:,localFluxBoundaryNodes]
        MDirichletNodes = MDirichletFull[localFluxBoundaryNodes][:,localDirichletBoundaryNodes]

        RFull = RT[:,TInd]
        RFluxNodes = RFull[localFluxBoundaryNodes]

        # Find which world Dirichlet nodes corresponds to the local Dirichlet nodes
        elementNodes = util.convertpCoordinateToIndex(NWorldCoarse, iWorldCoarse) + \
                       util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        elementDirichletBoundaryNodes = elementNodes[localDirichletBoundaryNodes]
        sigmaDirichletLocalized = sigmaDirichlet[elementDirichletBoundaryNodes]

        # Solve MFluxNodes*sigma = RFluxNodes - MDirichletNodes*sigmaDirichletLocalized
        bFluxNodes = RFluxNodes - MDirichletNodes*sigmaDirichletLocalized
        sigmaFlux = sparse.linalg.spsolve(MFluxNodes, bFluxNodes)

        sigmaFluxT[localFluxBoundaryNodes, TInd] = sigmaFlux
        
    return sigmaFluxT, sigmaDirichlet
