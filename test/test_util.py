import unittest
import numpy as np

from gridlod import util

class convert_TestCase(unittest.TestCase):
    def test_convertpCoordIndexToLinearIndex(self):
        N = np.array([100])
        coord = np.array([44])
        self.assertTrue(util.convertpCoordIndexToLinearIndex(N, coord) == 44)

        N = np.array([100, 200])
        coord = np.array([44, 55])
        self.assertTrue(util.convertpCoordIndexToLinearIndex(N, coord) == 44+55*101)

    def test_convertpLinearIndexToCoordIndex(self):
        N = np.array([100])
        ind = np.array([44])
        self.assertTrue(np.all(util.convertpLinearIndexToCoordIndex(N, ind) == [44]))

        N = np.array([100, 200])
        ind = 44+55*101
        self.assertTrue(np.all(util.convertpLinearIndexToCoordIndex(N, ind) == [44, 55]))
        
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
        self.assertTrue(np.all(util.interiorpIndexMap(N) == list(range(1,10))))

        N = np.array([2,2,2,2])
        self.assertTrue(util.interiorpIndexMap(N) == [1+3+9+27])

        N = np.array([2,2,3,2])
        self.assertTrue(np.all(util.interiorpIndexMap(N) == [1+3+9+36, 1+3+9+36+9]))

class boundarypIndexMap_TestCase(unittest.TestCase):
    def test_boundarypIndexMap(self):
        N = np.array([3,4,5])
        Np = np.prod(N+1)
        shouldBe = np.setdiff1d(np.arange(Np), util.interiorpIndexMap(N))
        self.assertTrue(np.all(util.boundarypIndexMapLarge(N) == shouldBe))
                     
        boundaryMap = np.array([[True, False],
                                [True, True],
                                [True, True]])
        shouldBe = np.setdiff1d(np.arange(Np), util.interiorpIndexMap(N))
        coords = util.pCoordinates(N)
        shouldBe = np.setdiff1d(shouldBe, np.where(np.logical_and(coords[:,0] == 1,
                                                   np.logical_and(coords[:,1] != 0,
                                                   np.logical_and(coords[:,1] != 1,
                                                   np.logical_and(coords[:,2] != 0,
                                                                  coords[:,2] != 1)))))[0])
        self.assertTrue(np.all(util.boundarypIndexMapLarge(N, boundaryMap) == shouldBe))
        
        boundaryMap = np.array([[True, True],
                                [False, True],
                                [True, True]])
        shouldBe = np.setdiff1d(np.arange(Np), util.interiorpIndexMap(N))
        shouldBe = np.setdiff1d(shouldBe, np.where(np.logical_and(coords[:,0] != 0,
                                                   np.logical_and(coords[:,0] != 1,
                                                   np.logical_and(coords[:,1] == 0,
                                                   np.logical_and(coords[:,2] != 0,
                                                                  coords[:,2] != 1)))))[0])
        self.assertTrue(np.all(util.boundarypIndexMapLarge(N, boundaryMap) == shouldBe))

        boundaryMap = np.array([[True, True],
                                [False, False],
                                [True, True]])
        shouldBe = np.setdiff1d(np.arange(Np), util.interiorpIndexMap(N))
        shouldBe = np.setdiff1d(shouldBe, np.where(np.logical_and(coords[:,0] != 0,
                                                   np.logical_and(coords[:,0] != 1,
                                                   np.logical_and(coords[:,1] == 0,
                                                   np.logical_and(coords[:,2] != 0,
                                                                  coords[:,2] != 1)))))[0])
        shouldBe = np.setdiff1d(shouldBe, np.where(np.logical_and(coords[:,0] != 0,
                                                   np.logical_and(coords[:,0] != 1,
                                                   np.logical_and(coords[:,1] == 1,
                                                   np.logical_and(coords[:,2] != 0,
                                                                  coords[:,2] != 1)))))[0])
        self.assertTrue(np.all(util.boundarypIndexMapLarge(N, boundaryMap) == shouldBe))

class extractPatchFine_TestCase(unittest.TestCase):
    def test_extractPatchFine(self):
        NCoarse = np.array([10])
        NCoarseElement = np.array([3])
        iPatchCoarse = np.array([2])
        NPatchCoarse = np.array([3])
        
        patchFineIndexMap = util.extractPatchFine(NCoarse, NCoarseElement, iPatchCoarse, NPatchCoarse,
                                                  extractElements=True)
        self.assertTrue(np.all(patchFineIndexMap == 6+np.arange(9)))

        patchFineIndexMap = util.extractPatchFine(NCoarse, NCoarseElement, iPatchCoarse, NPatchCoarse,
                                                  extractElements=False)
        self.assertTrue(np.all(patchFineIndexMap == 6+np.arange(10)))
        
class coordinate_TestCase(unittest.TestCase):
    def test_pCoordinates(self):
        NWorld = np.array([5])
        xp = util.pCoordinates(NWorld)
        self.assertTrue(np.allclose(xp.T - np.array([0., 1., 2., 3., 4., 5.])/5, 0))

        NWorld = np.array([9, 9, 9, 9])
        xp = util.pCoordinates(NWorld)
        ind = util.convertpCoordIndexToLinearIndex(NWorld, [7, 3, 6, 0])
        self.assertTrue(np.allclose(xp[ind] - np.array([7., 3., 6., 0.])/9., 0))
        
    def test_tCoordinates(self):
        NWorld = np.array([5])
        xt = util.tCoordinates(NWorld)
        self.assertTrue(np.isclose(np.max(np.abs(xt.T - np.array([1., 3., 5., 7., 9.])/10)), 0))

        NWorld = np.array([9, 9, 9, 9])
        xt = util.tCoordinates(NWorld)
        ind = util.convertpCoordIndexToLinearIndex(NWorld-1, [7, 3, 6, 0])
        self.assertTrue(np.allclose(xt[ind] - np.array([15., 7., 13., 1.])/18., 0))
        
if __name__ == '__main__':
    unittest.main()
