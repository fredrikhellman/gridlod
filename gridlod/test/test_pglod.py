import unittest
import numpy as np

from gridlod import pglod
from gridlod.world import World, Patch

class assembleBasisCorrectors_TestCase(unittest.TestCase):
    def test_fullPatch(self):
        NWorldCoarse = np.array([3,3])
        NCoarseElement = np.array([2,2])

        world = World(NWorldCoarse, NCoarseElement)
        k = 3
        
        patchT = [Patch(world, k, TInd) for TInd in range(world.NtCoarse)]
        basisCorrectorsListT = [[np.zeros(world.NpFine),
                                 np.zeros(world.NpFine),
                                 np.zeros(world.NpFine),
                                 np.zeros(world.NpFine)] for i in range(world.NtCoarse)]

        # Set first and last corrector to constant 1 and 2
        basisCorrectorsListT[0][0][:] = 1
        basisCorrectorsListT[-1][-1][:] = 2
        
        basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, basisCorrectorsListT)

        self.assertTrue(np.allclose(basisCorrectors.todense()[:,0], 1))
        self.assertTrue(np.allclose(basisCorrectors.todense()[:,-1], 2))

    def test_smallPatch(self):
        NWorldCoarse = np.array([3,3])
        NCoarseElement = np.array([2,2])

        world = World(NWorldCoarse, NCoarseElement)
        k = 0
        
        patchT = [Patch(world, k, TInd) for TInd in range(world.NtCoarse)]
        basisCorrectorsListT = [[np.zeros(patchT[0].NpFine),
                                 np.zeros(patchT[0].NpFine),
                                 np.zeros(patchT[0].NpFine),
                                 np.zeros(patchT[0].NpFine)] for i in range(world.NtCoarse)]

        # Set first and last corrector to constant 1 and 2
        basisCorrectorsListT[0][0][:] = 1
        basisCorrectorsListT[-1][-1][:] = 2
        
        basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, basisCorrectorsListT)

        firstBasisCorrectorShouldBe = np.array([1, 1, 1, 0, 0, 0, 0,
                                                1, 1, 1, 0, 0, 0, 0,
                                                1, 1, 1, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0,
                                                0, 0, 0, 0, 0, 0, 0])

        lastBasisCorrectorShouldBe = np.array([0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 0, 0, 0,
                                               0, 0, 0, 0, 2, 2, 2,
                                               0, 0, 0, 0, 2, 2, 2,
                                               0, 0, 0, 0, 2, 2, 2])

        self.assertTrue(np.allclose(basisCorrectors.todense()[:,0].squeeze(), firstBasisCorrectorShouldBe))
        self.assertTrue(np.allclose(basisCorrectors.todense()[:,-1].squeeze(), lastBasisCorrectorShouldBe))
        
if __name__ == '__main__':
    unittest.main()
