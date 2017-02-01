import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from world import World
import flux
import fem
import util

class computeBoundaryFlux_TestCase(unittest.TestCase):

    def test_computeBoundaryFlux_1d(self):
        NWorldCoarse = np.array([10])
        NCoarseElement = np.array([1])
        boundaryConditions = np.array([[1, 0]])
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NWorldFine = NWorldCoarse*NCoarseElement
        
        coords = util.pCoordinates(NWorldFine)

        Np = np.prod(NWorldCoarse+1)
        
        fFull = -coords[:,0]
        AFull = fem.assemblePatchMatrix(NWorldCoarse, world.ALocCoarse)
        A = AFull[:-1][:,:-1]
        MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
        bFull = MFull*fFull
        b = bFull[:-1]
        
        u = sparse.linalg.spsolve(A, b)
        uFull = np.zeros(Np)
        uFull[:-1] = u

        ROmega = AFull*uFull - bFull
    
        sigma = flux.computeBoundaryFlux(world, ROmega)
        self.assertTrue(np.size(sigma) == Np)
        self.assertTrue(np.isclose(sigma[-1], 0.5))
        self.assertTrue(np.isclose(sigma[-1], NWorldCoarse[0]*(uFull[-1]-uFull[-2]), atol=1./NWorldCoarse[0]))
        
    def test_computeBoundaryFlux_2d(self):
        NWorldCoarse = np.array([10, 10])
        NCoarseElement = np.array([1, 1])
        boundaryConditions = np.array([[0, 0], [1, 1]])
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NWorldFine = NWorldCoarse*NCoarseElement
        
        coords = util.pCoordinates(NWorldFine)
        uCoarse = coords[:,0] + (coords[:,0]>0.5)*coords[:,0]

        Np = np.prod(NWorldCoarse+1)
                
        ACoarseFull = fem.assemblePatchMatrix(NWorldCoarse, world.ALocCoarse)

        ROmega = ACoarseFull*uCoarse
    
        sigma = flux.computeBoundaryFlux(world, ROmega)

        boundaryMap = boundaryConditions==0
        dirichletNodes = util.boundarypIndexMap(NWorldCoarse, boundaryMap)
        otherNodes = np.setdiff1d(np.arange(Np), dirichletNodes)
        self.assertTrue(np.allclose(sigma[dirichletNodes][::2], -1.))
        self.assertTrue(np.allclose(sigma[dirichletNodes][1::2], 2.))
        self.assertTrue(np.allclose(sigma[otherNodes], 0.))
    
    def test_computeCoarseElementFlux_1d(self):
        NWorldCoarse = np.array([10])
        NCoarseElement = np.array([1])
        boundaryConditions = np.array([[1, 0]])
        d = 1
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NWorldFine = NWorldCoarse*NCoarseElement
        
        coords = util.pCoordinates(NWorldFine)

        Np = np.prod(NWorldCoarse+1)
        Nt = np.prod(NWorldCoarse)
        
        fFull = -coords[:,0]
        AFull = fem.assemblePatchMatrix(NWorldCoarse, world.ALocCoarse)
        A = AFull[:-1][:,:-1]
        MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
        bFull = MFull*fFull
        b = bFull[:-1]
        
        u = sparse.linalg.spsolve(A, b)
        uFull = np.zeros(Np)
        uFull[:-1] = u

        ROmega = AFull*uFull - bFull
        RT = np.zeros([2**d, Nt])
        for T in np.arange(Nt):
            RT[:,T] = np.dot(world.ALocCoarse, np.array([uFull[T], uFull[T+1]])) - \
                      np.dot(world.MLocCoarse, np.array([fFull[T], fFull[T+1]]))
    
        sigmaFluxT, sigmaDirichlet = flux.computeCoarseElementFlux(world, ROmega, RT)

        print sigmaFluxT
        self.assertTrue(np.allclose(sigmaFluxT[0,1:], -sigmaFluxT[1,0:-1]))
        self.assertTrue(np.allclose(sigmaFluxT[0,:]+sigmaFluxT[1,:], -0.5*(fFull[:-1]+fFull[1:])/Nt))
        
if __name__ == '__main__':
    unittest.main()
