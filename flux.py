import numpy as np
import scipy.sparse as sparse

from world import World
import util
import fem
import linalg

class ConservativeFluxComputer:
    def __init__(self, world, R):
        '''
        world 

        R      
               a function that, given a subdomain omega,
               returns the residual functional:
                   R(omega)(v) = a(u, v)_omega - (f, v)_omega
               for v in the full FE space of fineness given 
               in NCoarseElement.'''
        self.world = world
        self.R = R

    def computeBoundaryFlux(self):
        world = self.world
        R = self.R

        NWorldFine = world.NWorldCoarse*world.NCoarseElement
        NWorldCoarse = world.NWorldCoarse
        NCoarseElement = world.NCoarseElement
        
        boundaryMap = world.boundaryConditions==0
        dirichletNodes = util.boundarypIndexMap(NWorldCoarse, boundaryMap)

        MLocGetter = fem.localBoundaryMassMatrixGetter(NWorldCoarse)
        MDirichletFull = fem.assemblePatchBoundaryMatrix(NWorldCoarse,
                                                         MLocGetter,
                                                         boundaryMap=boundaryMap)

        ROmega = R(np.zeros_like(NWorldCoarse), NWorldCoarse)

        # Solve MDirichletNodes * sigmaDirichletNodes = RDirichletNodes
        MDirichletNodes = MDirichletFull[dirichletNodes][:,dirichletNodes]
        RDiricheletNodes = ROmega[dirichletNodes]

        omegaDirichletNodes = sparse.linalg.spsolve(MDirichletNodes, RDiricheletNodes)

        return omegaDirichletNodes, dirichletNodes
