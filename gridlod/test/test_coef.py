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

class coefficientCoarseFactor_TestCase(unittest.TestCase):
    def test_coefficientCoarseFactor(self):
        aBase = np.array([1,2,3,3,2,1,4,5,6,6,5,4])
        rCoarse = np.array([10, 20, 100, 200])
        NPatchCoarse = np.array([2, 2])
        NCoarseElement = np.array([3, 1])

        coefFirst = coef.CoefficientCoarseFactor(NPatchCoarse, NCoarseElement, aBase, rCoarse)

        aFine = coefFirst.aFine
        self.assertTrue(np.all(aFine == [10, 20, 30, 60, 40, 20, 400, 500, 600, 1200, 1000, 800]))

        iSubPatchCoarse = np.array([0, 1])
        NSubPatchCoarse = np.array([1, 1])
        coefLocalized = coefFirst.localize(iSubPatchCoarse, NSubPatchCoarse)

        aFineLocalized = coefLocalized.aFine
        self.assertTrue(np.all(aFineLocalized == [400, 500, 600]))
        
        coefLocalizeFirst = coef.CoefficientCoarseFactor(NPatchCoarse, NCoarseElement, aBase, rCoarse)
        coefLocalizedFirst = coefLocalizeFirst.localize(iSubPatchCoarse, NSubPatchCoarse)
        aFineLocalizedFirst = coefLocalizedFirst.aFine

        self.assertTrue(np.all(aFineLocalized ==  aFineLocalizedFirst))

        rCoarseLocalizedRecovered = coefLocalized.rCoarse
        self.assertTrue(rCoarseLocalizedRecovered == 100)
        
if __name__ == '__main__':
    unittest.main()
