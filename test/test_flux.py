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

        def ROmega(iPatchCoarse, NPatchCoarse):
            self.assertTrue(np.max(np.abs(iPatchCoarse)) == 0)
            self.assertTrue(np.all(NPatchCoarse == NWorldCoarse))

            return AFull*uFull - bFull
    
        fluxComp = flux.ConservativeFluxComputer(world, ROmega)
        omega, omegaNodes = fluxComp.computeBoundaryFlux()
        self.assertTrue(np.size(omega) == 1)
        self.assertTrue(np.isclose(omega, 0.5))
        self.assertTrue(np.isclose(omega, NWorldCoarse[0]*(uFull[-1]-uFull[-2]), atol=1./NWorldCoarse[0]))
        
    def test_computeBoundaryFlux_2d(self):
        NWorldCoarse = np.array([10, 10])
        NCoarseElement = np.array([1, 1])
        boundaryConditions = np.array([[0, 0], [1, 1]])
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NWorldFine = NWorldCoarse*NCoarseElement
        
        coords = util.pCoordinates(NWorldFine)
        uCoarse = coords[:,0] + (coords[:,0]>0.5)*coords[:,0]
        
        def ROmega(iPatchCoarse, NPatchCoarse):
            self.assertTrue(np.max(np.abs(iPatchCoarse)) == 0)
            self.assertTrue(np.all(NPatchCoarse == NWorldCoarse))
            
            ACoarseFull = fem.assemblePatchMatrix(NPatchCoarse, world.ALocCoarse)
            return ACoarseFull*uCoarse
    
        fluxComp = flux.ConservativeFluxComputer(world, ROmega)
        omega, omegaNodes = fluxComp.computeBoundaryFlux()
        self.assertTrue(np.allclose(omega[::2], -1.))
        self.assertTrue(np.allclose(omega[1::2], 2.))
    
if __name__ == '__main__':
    unittest.main()
