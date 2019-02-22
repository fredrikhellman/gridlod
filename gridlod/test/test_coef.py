import unittest
import numpy as np

from gridlod import coef
from gridlod.world import World, Patch

class localizeCoefficient_TestCase(unittest.TestCase):
    def test_localizeCoefficient(self):
        NWorldCoarse = np.array([2, 3])
        NCoarseElement = np.array([2, 1])
        world = World(NWorldCoarse, NCoarseElement)
        patch = Patch(world, 1, 0)
        aFine = np.array([1, 2, 3, 4,
                          10, 20, 30, 40,
                          100, 200, 300, 400])
        aFineLocalized = coef.localizeCoefficient(patch, aFine)
        print(aFineLocalized)
        self.assertTrue(np.all(aFineLocalized == aFine[:8]))

if __name__ == '__main__':
    unittest.main()
