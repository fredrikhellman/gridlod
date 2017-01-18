import unittest
import numpy as np

import pg
import world
import interp
import coef

class PetrovGalerkinLOD_TestCase(unittest.TestCase):
    def test_trivials(self):
        NWorldCoarse = np.array([1,2,3])
        NCoarseElement = np.array([2,3,4])
        NFine = NWorldCoarse*NCoarseElement
        NtFine = np.prod(NFine)
        
        wrld = world.World(NWorldCoarse, NCoarseElement)

        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement)
        
        k = 1
        
        pglod = pg.PetrovGalerkinLOD(wrld, k, IPatchGenerator, 0)

        aBase = np.ones(NtFine)
        pglod.updateCorrectors(coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase))
        pglod.assembleStiffnessMatrix()

        # LAGG TILL TESTER
        
if __name__ == '__main__':
    unittest.main()
