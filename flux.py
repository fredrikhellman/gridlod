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
    
    asInPaper=False
    if asInPaper:
        # Romega needs to be reintroduced as argument if this is enabled
        sigmaDirichlet = computeBoundaryFlux(world, ROmega)
    
    worldDirichletBoundaryMap = world.boundaryConditions==0
    MLocGetter = fem.localBoundaryMassMatrixGetter(NWorldCoarse)

    NTInds = np.size(TInds)
    sigmaFluxT = np.zeros([2**d, NTInds])
    nodeFluxT = np.zeros([2**d, NTInds])

    for TInd in TInds:
        print ',',
        iWorldCoarse = util.convertpIndexToCoordinate(NWorldCoarse-1, TInd)
        boundary0 = iWorldCoarse==0
        boundary1 = (iWorldCoarse + 1) == NWorldCoarse

        # Much of this is equal from iteration to iteration, can be sped up if needed
        localBoundaryMap = np.column_stack([boundary0, boundary1])
        localBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                    localBoundaryMap)
        if asInPaper:
            localFluxBoundaryMap = np.logical_not(localBoundaryMap)
        else:
            localFluxBoundaryMap = np.logical_not(np.logical_and(localBoundaryMap, world.boundaryConditions==1))

        localFluxBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                        localFluxBoundaryMap)

        MFluxFull = fem.assemblePatchBoundaryMatrix(np.ones_like(NWorldCoarse),
                                                    MLocGetter,
                                                    boundaryMap=localFluxBoundaryMap).todense()
        MFluxNodes = MFluxFull[localFluxBoundaryNodes][:,localFluxBoundaryNodes]

        
        RFull = RT[:,TInd]
        RFluxNodes = RFull[localFluxBoundaryNodes]

        # Solve MFluxNodes*sigma = RFluxNodes - MDirichletNodes*sigmaDirichletLocalized
        bFluxNodes = RFluxNodes
        
        if asInPaper:
            localDirichletBoundaryMap = np.logical_and(localBoundaryMap, worldDirichletBoundaryMap)
            localDirichletBoundaryNodes = util.boundarypIndexMap(np.ones_like(NWorldCoarse),
                                                             localDirichletBoundaryMap)
            MDirichletFull = fem.assemblePatchBoundaryMatrix(np.ones_like(NWorldCoarse),
                                                             MLocGetter,
                                                             boundaryMap=localDirichletBoundaryMap)
            MDirichletNodes = MDirichletFull[localFluxBoundaryNodes][:,localDirichletBoundaryNodes]
            elementNodes = util.convertpCoordinateToIndex(NWorldCoarse, iWorldCoarse) + \
                           util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
            elementDirichletBoundaryNodes = elementNodes[localDirichletBoundaryNodes]
            sigmaDirichletLocalized = sigmaDirichlet[elementDirichletBoundaryNodes]
            bFluxNodes -= MDirichletNodes*sigmaDirichletLocalized
            
        sigmaFlux = np.linalg.solve(MFluxNodes, bFluxNodes)
        sigmaFluxT[localFluxBoundaryNodes, TInd] = sigmaFlux
        nodeFlux = np.dot(MFluxNodes, sigmaFlux)
        nodeFluxT[localFluxBoundaryNodes, TInd] = nodeFlux
        
    return sigmaFluxT, nodeFluxT
