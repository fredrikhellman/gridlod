import unittest
import numpy as np

from gridlod import lod_flux, fem, interp, util, coef, world
from gridlod.world import World

class lod_flux_TestCase(unittest.TestCase):
    def test_computeCoarseErrorIndicatorFlux(self):
        NWorldCoarse = np.array([7, 7], dtype='int64')
        NCoarseElement = np.array([10, 10], dtype='int64')
        NWorldFine = NWorldCoarse*NCoarseElement
        NpWorldFine = np.prod(NWorldFine+1)
        NpWorldCoarse = np.prod(NWorldCoarse+1)
        NtWorldFine = np.prod(NWorldCoarse*NCoarseElement)
        NtWorldCoarse = np.prod(NWorldCoarse)

        np.random.seed(0)

        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        aBase = np.exp(np.random.rand(NtWorldFine))
        k = np.max(NWorldCoarse)
        iElementWorldCoarse = np.array([3,3])

        rCoarseFirst = 1+3*np.random.rand(NtWorldCoarse)
        coefFirst = coef.CoefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarseFirst)
        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement)
        ec = lod_flux.CoarseBasisElementCorrectorFlux(world, k, iElementWorldCoarse, IPatchGenerator)
        ec.computeCorrectors(coefFirst)
        ec.computeCoarseQuantities()

        # If both rCoarseFirst and rCoarseSecond are equal, the error indicator should be zero
        rCoarseSecond = np.array(rCoarseFirst)
        self.assertTrue(np.isclose(ec.computeCoarseErrorIndicatorFlux(rCoarseSecond), 0))

        coefSecond = coef.CoefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarseSecond)
        self.assertTrue(np.isclose(ec.computeErrorIndicatorFine(coefSecond), 0))
        
        # If rCoarseSecond is not rCoarseFirst, the error indicator should not be zero
        rCoarseSecond = 2*np.array(rCoarseFirst)
        self.assertTrue(ec.computeCoarseErrorIndicatorFlux(rCoarseSecond) >= 0.1)

        coefSecond = coef.CoefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarseSecond)
        self.assertTrue(ec.computeErrorIndicatorFine(coefSecond) >= 0.1)

        # Fine should be smaller than coarse estimate
        self.assertTrue(ec.computeErrorIndicatorFine(coefSecond) < ec.computeCoarseErrorIndicatorFlux(rCoarseSecond))

        # If rCoarseSecond is different in the element itself, the error
        # indicator should be large
        elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NWorldCoarse-1, iElementWorldCoarse)
        rCoarseSecond = np.array(rCoarseFirst)
        rCoarseSecond[elementCoarseIndex] *= 2
        saveForNextTest = ec.computeCoarseErrorIndicatorFlux(rCoarseSecond)
        self.assertTrue(saveForNextTest >= 0.1)

        coefSecond = coef.CoefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarseSecond)
        fineResult = ec.computeErrorIndicatorFine(coefSecond)
        self.assertTrue(fineResult >= 0.1)
        self.assertTrue(ec.computeErrorIndicatorFine(coefSecond) < ec.computeCoarseErrorIndicatorFlux(rCoarseSecond))

        # A difference in the perifery should be smaller than in the center
        rCoarseSecond = np.array(rCoarseFirst)
        rCoarseSecond[0] *= 2
        self.assertTrue(saveForNextTest > ec.computeCoarseErrorIndicatorFlux(rCoarseSecond))

        # Again, but closer
        rCoarseSecond = np.array(rCoarseFirst)
        rCoarseSecond[elementCoarseIndex-1] *= 2
        self.assertTrue(saveForNextTest > ec.computeCoarseErrorIndicatorFlux(rCoarseSecond))

if __name__ == '__main__':
    unittest.main()
