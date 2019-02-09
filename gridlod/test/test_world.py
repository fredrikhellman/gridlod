import unittest
import numpy as np

from gridlod.world import World, Patch

class Patch_TestCase(unittest.TestCase):
    def test_Patch(self):
        NWorldCoarse = np.array([4, 4])
        NCoarseElement = np.array([2,2])
        world = World(NWorldCoarse, NCoarseElement)

        k = 1

        TInd = 0
        patch = Patch(world, k, TInd)
        self.assertTrue(np.all(patch.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(patch.iElementPatchCoarse == [0, 0]))
        self.assertTrue(np.all(patch.iPatchWorldCoarse == [0, 0]))

        TInd = 12
        patch = Patch(world, k, TInd)
        self.assertTrue(np.all(patch.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(patch.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(patch.iPatchWorldCoarse == [0, 2]))

        TInd = 8
        patch = Patch(world, k, TInd)
        self.assertTrue(np.all(patch.NPatchCoarse == [2, 3]))
        self.assertTrue(np.all(patch.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(patch.iPatchWorldCoarse == [0, 1]))

        TInd = 9
        patch = Patch(world, k, TInd)
        self.assertTrue(np.all(patch.NPatchCoarse == [3, 3]))
        self.assertTrue(np.all(patch.iElementPatchCoarse == [1, 1]))
        self.assertTrue(np.all(patch.iPatchWorldCoarse == [0, 1]))
        
if __name__ == '__main__':
    unittest.main()
