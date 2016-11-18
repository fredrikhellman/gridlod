import unittest
import numpy as np

import util

class pIndexMap_TestCase(unittest.TestCase):
    def test_trivials1d(self):
        # 1D
        NFrom = np.array([1])
        NTo = np.array([1])
        NStep = np.array([1])
        self.assertTrue(np.all(util.pIndexMap(NFrom, NTo, NStep) == [0, 1]))

        NFrom = np.array([0])
        NTo = np.array([1])
        NStep = np.array([1])
        self.assertTrue(np.all(util.pIndexMap(NFrom, NTo, NStep) == [0]))

    def test_trivials3d(self):
        # 3D
        NFrom = np.array([1, 1, 1])
        NTo = np.array([2, 2, 2])
        NStep = np.array([0, 0, 0])
        self.assertTrue(np.all(util.pIndexMap(NFrom, NTo, NStep) == 0))

    def test_examples(self):
        NFrom = np.array([1, 1, 1])
        NTo = np.array([1, 9, 1])
        NStep = np.array([1, 1, 1])
        self.assertTrue(np.all(util.pIndexMap(NFrom, NTo, NStep) == [0, 1, 2, 3, 20, 21, 22, 23]))

        NFrom = np.array([1, 1, 1])
        NTo = np.array([9, 1, 1])
        NStep = np.array([9, 1, 1])
        self.assertTrue(np.all(util.pIndexMap(NFrom, NTo, NStep) == [0, 9, 10, 19, 20, 29, 30, 39]))

class numNeighboringElements_TestCase(unittest.TestCase):
    def test_trivials(self):
        iPatch = np.array([1, 1, 1])
        NPatch = np.array([1, 1, 1])
        NWorld = np.array([3, 3, 3])
        self.assertTrue(np.all(util.numNeighboringElements(iPatch, NPatch, NWorld) == [8]*8))

        iPatch = np.array([0, 0, 0])
        NPatch = np.array([1, 1, 1])
        NWorld = np.array([1, 1, 1])
        self.assertTrue(np.all(util.numNeighboringElements(iPatch, NPatch, NWorld) == [1]*8))
        
    def test_examples(self):
        iPatch = np.array([0, 1, 0])
        NPatch = np.array([1, 1, 1])
        NWorld = np.array([2, 2, 2])
        self.assertTrue(np.all(util.numNeighboringElements(iPatch, NPatch, NWorld) == [2, 4, 1, 2, 4, 8, 2, 4]))

class interiorpIndexMap_TestCase(unittest.TestCase):
    def test_trivials(self):
        N = np.array([1])
        self.assertTrue(np.size(util.interiorpIndexMap(N)) == 0)

        N = np.array([2])
        self.assertTrue(np.all(util.interiorpIndexMap(N) == [1]))

    def test_examples(self):
        N = np.array([10])
        self.assertTrue(np.all(util.interiorpIndexMap(N) == range(1,10)))

        N = np.array([2,2,2,2])
        self.assertTrue(util.interiorpIndexMap(N) == [1+3+9+27])

        N = np.array([2,2,3,2])
        self.assertTrue(np.all(util.interiorpIndexMap(N) == [1+3+9+36, 1+3+9+36+9]))

class boundarypIndexMap_TestCase(unittest.TestCase):
    def test_boundarypIndexMap(self):
        N = np.array([3,4,5])
        self.assertTrue(np.all(util.boundarypIndexMapSmall(N) ==
                               util.boundarypIndexMapLarge(N)))

        N = np.array([30,10,20])
        self.assertTrue(np.all(util.boundarypIndexMapSmall(N) ==
                               util.boundarypIndexMapLarge(N)))
        
if __name__ == '__main__':
    unittest.main()
