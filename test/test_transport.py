import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from gridlod import util, transport

class computeElementFaceVelocity_TestCase(unittest.TestCase):
    def test_computeElementFaceVelocity_1d(self):
        NPatchCoarse = np.array([4])
        NCoarseElement = np.array([10])
        NPatchFine = NPatchCoarse*NCoarseElement

        NtFine = np.prod(NPatchFine)
        NpFine = np.prod(NPatchFine+1)

        aPatch = np.ones(NtFine)
        uPatch = np.ones(NpFine)
        
        velocityTF = transport.computeElementFaceVelocity(NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.all(velocityTF.shape == (4, 2)))
        self.assertTrue(np.all(velocityTF == 0))

        uPatch = np.linspace(0, 1, NpFine)
        
        velocityTF = transport.computeElementFaceVelocity(NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(velocityTF[:,0], -1) and np.allclose(velocityTF[:,1], 1))

        aPatch[0:10] = 5
        aPatch[10:20] = 100
        aPatch[20:30] = 3
        aPatch[30:40] = 2
        velocityTF = transport.computeElementFaceVelocity(NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(velocityTF, [[-5, 5],
                                         [-100, 100],
                                         [-3, 3],
                                         [-2, 2]]))

        aPatch[0] = 10
        aPatch[9] = 20
        aPatch[10] = 5
        aPatch[19] = 6
        velocityTF = transport.computeElementFaceVelocity(NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(velocityTF, [[-10, 20],
                                         [-5, 6],
                                         [-3, 3],
                                         [-2, 2]]))

        uPatch[10] = 1./40
        uPatch[9] = 2./40
        uPatch[11] = 0
        velocityTF = transport.computeElementFaceVelocity(NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(velocityTF, [[-10, -20],
                                         [5, 6],
                                         [-3, 3],
                                         [-2, 2]]))
        
    def test_computeElementFaceVelocity_2d(self):
        NPatchCoarse = np.array([4, 6])
        NCoarseElement = np.array([10, 20])
        NPatchFine = NPatchCoarse*NCoarseElement

        NtFine = np.prod(NPatchFine)
        NpFine = np.prod(NPatchFine+1)

        aPatch = np.ones(NtFine)
        uPatch = np.ones(NpFine)
        
        velocityTF = transport.computeElementFaceVelocity(NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.all(velocityTF.shape == (24, 4)))
        self.assertTrue(np.all(velocityTF == 0))

        x = util.pCoordinates(NPatchFine)[:,0]
        y = util.pCoordinates(NPatchFine)[:,1]
        uPatch = x

        xFaceArea = 1./NPatchCoarse[1]
        velocityTF = transport.computeElementFaceVelocity(NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(velocityTF[:,0], -xFaceArea) and np.allclose(velocityTF[:,1], xFaceArea))

        uPatch = x+y

        xFaceArea = 1./NPatchCoarse[1]
        yFaceArea = 1./NPatchCoarse[0]
        velocityTF = transport.computeElementFaceVelocity(NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(velocityTF[:,0], -xFaceArea) and np.allclose(velocityTF[:,1], xFaceArea))
        self.assertTrue(np.allclose(velocityTF[:,2], -yFaceArea) and np.allclose(velocityTF[:,3], yFaceArea))

        aPatch = 10*aPatch
        velocityTF = transport.computeElementFaceVelocity(NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(velocityTF[:,0], -10*xFaceArea) and np.allclose(velocityTF[:,1], 10*xFaceArea))
        self.assertTrue(np.allclose(velocityTF[:,2], -10*yFaceArea) and np.allclose(velocityTF[:,3], 10*yFaceArea))
        
class computeAverageFaceVelocity_TestCase(unittest.TestCase):
    def test_computeAverageFaceVelocity_1d(self):
        NWorldCoarse = np.array([4])
        velocityTF = np.array([[3., -4],
                       [6, -10],
                       [10, -1],
                       [2, 0]])
        
        avgVelocityTF = transport.computeAverageFaceVelocity(NWorldCoarse, velocityTF)
        self.assertTrue(np.allclose(avgVelocityTF, [[3, -5],
                                                    [5, -10],
                                                    [10, -1.5],
                                                    [1.5, 0]]))

    def test_computeAverageFaceVelocity_2d(self):
        NWorldCoarse = np.array([2, 2])
        velocityTF = np.array([[3., -6, 2, -2],
                       [4, -4, 3, -3],
                       [4, -1, 4, 1],
                       [-1, 10, 3, 0]])
        
        avgVelocityTF = transport.computeAverageFaceVelocity(NWorldCoarse, velocityTF)
        print avgVelocityTF
        self.assertTrue(np.allclose(avgVelocityTF, [[3., -5, 2, -3],
                                                    [5, -4, 3, -3],
                                                    [4, 0, 3, 1],
                                                    [0, 10, 3, 0]]))
        

class computeUpwindSaturation_TestCase(unittest.TestCase):
    def test_computeUpwindSaturation_1d(self):
        NWorldCoarse = np.array([4])
        boundarys = np.array([[33., 100.]])

        sT = np.array([1, 2, 3, 4])
        velocityTF = np.array([[-1, 1],
                               [-1, -4],
                               [4, -2],
                               [2, 2]])

        sTF = transport.computeUpwindSaturation(NWorldCoarse, boundarys, sT, velocityTF)
        self.assertTrue(np.allclose(sTF, [[33., 1],
                                          [1, 3],
                                          [3, 4],
                                          [4, 4]]))
        velocityTF = np.array([[1, 1],
                               [-1, -4],
                               [4, -2],
                               [2, -2]])

        sTF = transport.computeUpwindSaturation(NWorldCoarse, boundarys, sT, velocityTF)
        self.assertTrue(np.allclose(sTF, [[1., 1],
                                          [1, 3],
                                          [3, 4],
                                          [4, 100]]))

    def test_computeUpwindSaturation_2d(self):
        NWorldCoarse = np.array([2, 3])
        boundarys = np.array([[33.,   100.],
                              [1000., 2000.]])

        sT = np.array([1, 2, 3, 4, 5, 6])
        velocityTF = np.array([[-1,  1, 2, -3],
                               [-1,  1, -1, 2],
                               [2,   2, 3, -3],
                               [-2, -1, -2, 2],
                               [2,   2, 3, -1],
                               [-2, 10, -2, 2]])

        sTF = transport.computeUpwindSaturation(NWorldCoarse, boundarys, sT, velocityTF)
        self.assertTrue(np.allclose(sTF, [[33., 1,    1,    3],
                                          [1,   2,    1000, 2],
                                          [3,   3,    3,    5],
                                          [3,   100., 2,    4],
                                          [5,   5,    5,    2000.],
                                          [5,   6.,   4,    6]]))
        
if __name__ == '__main__':
    #import cProfile
    #command = """unittest.main()"""
    #cProfile.runctx( command, globals(), locals(), filename="test_pg.profile" )
    unittest.main()
