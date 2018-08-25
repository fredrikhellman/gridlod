import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from gridlod import util, transport, world
from gridlod.world import World

class computeElementFaceFlux_TestCase(unittest.TestCase):
    def test_computeElementFaceFlux_1d(self):
        NPatchCoarse = np.array([4])
        NCoarseElement = np.array([10])
        NPatchFine = NPatchCoarse*NCoarseElement

        NtFine = np.prod(NPatchFine)
        NpFine = np.prod(NPatchFine+1)

        aPatch = np.ones(NtFine)
        uPatch = np.ones(NpFine)
        
        fluxTF = transport.computeElementFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.all(fluxTF.shape == (4, 2)))
        self.assertTrue(np.all(fluxTF == 0))

        uPatch = np.linspace(0, 1, NpFine)
        
        fluxTF = transport.computeElementFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], 1) and np.allclose(fluxTF[:,1], -1))

        aPatch[0:10] = 5
        aPatch[10:20] = 100
        aPatch[20:30] = 3
        aPatch[30:40] = 2
        fluxTF = transport.computeElementFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF, [[5, -5],
                                         [100, -100],
                                         [3, -3],
                                         [2, -2]]))

        aPatch[0] = 10
        aPatch[9] = 20
        aPatch[10] = 5
        aPatch[19] = 6
        fluxTF = transport.computeElementFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF, [[10, -20],
                                         [5, -6],
                                         [3, -3],
                                         [2, -2]]))

        uPatch[10] = 1./40
        uPatch[9] = 2./40
        uPatch[11] = 0
        fluxTF = transport.computeElementFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF, [[10, 20],
                                         [-5, -6],
                                         [3, -3],
                                         [2, -2]]))
        
        fluxTF = transport.computeElementFaceFlux(10*NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF/10, [[10, 20],
                                                    [-5, -6],
                                                    [3, -3],
                                                    [2, -2]]))

    def test_computeElementFaceFlux_2d(self):
        NPatchCoarse = np.array([4, 6])
        NCoarseElement = np.array([10, 20])
        NPatchFine = NPatchCoarse*NCoarseElement

        NtFine = np.prod(NPatchFine)
        NpFine = np.prod(NPatchFine+1)

        aPatch = np.ones(NtFine)
        uPatch = np.ones(NpFine)
        
        fluxTF = transport.computeElementFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.all(fluxTF.shape == (24, 4)))
        self.assertTrue(np.all(fluxTF == 0))

        x = util.pCoordinates(NPatchFine)[:,0]
        y = util.pCoordinates(NPatchFine)[:,1]
        uPatch = x

        xFaceArea = 1./NPatchCoarse[1]
        fluxTF = transport.computeElementFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], xFaceArea) and np.allclose(fluxTF[:,1], -xFaceArea))

        uPatch = x+y

        xFaceArea = 1./NPatchCoarse[1]
        yFaceArea = 1./NPatchCoarse[0]
        fluxTF = transport.computeElementFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], xFaceArea) and np.allclose(fluxTF[:,1], -xFaceArea))
        self.assertTrue(np.allclose(fluxTF[:,2], yFaceArea) and np.allclose(fluxTF[:,3], -yFaceArea))

        aPatch = 10*aPatch
        fluxTF = transport.computeElementFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], 10*xFaceArea) and np.allclose(fluxTF[:,1], -10*xFaceArea))
        self.assertTrue(np.allclose(fluxTF[:,2], 10*yFaceArea) and np.allclose(fluxTF[:,3], -10*yFaceArea))

        aPatch = np.ones(NtFine)
        NWorldCoarse = np.array(NPatchCoarse)
        NWorldCoarse[0] = 10*NWorldCoarse[0]
        fluxTF = transport.computeElementFaceFlux(NWorldCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], 10*xFaceArea) and np.allclose(fluxTF[:,1], -10*xFaceArea))
        self.assertTrue(np.allclose(fluxTF[:,2], 0.1*yFaceArea) and np.allclose(fluxTF[:,3], -0.1*yFaceArea))
        
class computeAverageFaceFlux_TestCase(unittest.TestCase):
    def test_computeAverageFaceFlux_1d(self):
        NWorldCoarse = np.array([4])
        fluxTF = np.array([[3., -4],
                       [6, -10],
                       [10, -1],
                       [2, 0]])
        
        avgFluxTF = transport.computeAverageFaceFlux(NWorldCoarse, fluxTF)
        self.assertTrue(np.allclose(avgFluxTF, [[3, -5],
                                                    [5, -10],
                                                    [10, -1.5],
                                                    [1.5, 0]]))

    def test_computeAverageFaceFlux_2d(self):
        NWorldCoarse = np.array([2, 2])
        fluxTF = np.array([[3., -6, 2, -2],
                       [4, -4, 3, -3],
                       [4, -1, 4, 1],
                       [-1, 10, 3, 0]])
        
        avgFluxTF = transport.computeAverageFaceFlux(NWorldCoarse, fluxTF)
        self.assertTrue(np.allclose(avgFluxTF, [[3., -5, 2, -3],
                                                    [5, -4, 3, -3],
                                                    [4, 0, 3, 1],
                                                    [0, 10, 3, 0]]))
        

class computeUpwindSaturation_TestCase(unittest.TestCase):
    def test_computeUpwindSaturation_1d(self):
        NWorldCoarse = np.array([4])
        boundarys = np.array([[33., 100.]])

        sT = np.array([1, 2, 3, 4])
        fluxTF = np.array([[-1, 1],
                               [-1, -4],
                               [4, -2],
                               [2, 2]])

        sTF = transport.computeUpwindSaturation(NWorldCoarse, boundarys, sT, fluxTF)
        self.assertTrue(np.allclose(sTF, [[33., 1],
                                          [1, 3],
                                          [3, 4],
                                          [4, 4]]))
        fluxTF = np.array([[1, 1],
                               [-1, -4],
                               [4, -2],
                               [2, -2]])

        sTF = transport.computeUpwindSaturation(NWorldCoarse, boundarys, sT, fluxTF)
        self.assertTrue(np.allclose(sTF, [[1., 1],
                                          [1, 3],
                                          [3, 4],
                                          [4, 100]]))

    def test_computeUpwindSaturation_2d(self):
        NWorldCoarse = np.array([2, 3])
        boundarys = np.array([[33.,   100.],
                              [1000., 2000.]])

        sT = np.array([1, 2, 3, 4, 5, 6])
        fluxTF = np.array([[-1,  1, 2, -3],
                               [-1,  1, -1, 2],
                               [2,   2, 3, -3],
                               [-2, -1, -2, 2],
                               [2,   2, 3, -1],
                               [-2, 10, -2, 2]])

        sTF = transport.computeUpwindSaturation(NWorldCoarse, boundarys, sT, fluxTF)
        self.assertTrue(np.allclose(sTF, [[33., 1,    1,    3],
                                          [1,   2,    1000, 2],
                                          [3,   3,    3,    5],
                                          [3,   100., 2,    4],
                                          [5,   5,    5,    2000.],
                                          [5,   6.,   4,    6]]))
        
class computeHarmonicMeanAverageFaceFlux_TestCase(unittest.TestCase):
    def test_harmonicMeanOverFaces_1d(self):
        NWorldCoarse = np.array([4])
        NCoarseElement = np.array([3])
        aWorld = np.array([1e0, np.nan, 1e1,
                           1e2, np.nan, 1e3,
                           1e4, np.nan, 1e5,
                           1e6, np.nan, 1e7])

        a0Faces = transport.harmonicMeanOverFaces(NWorldCoarse, NCoarseElement, 0, 0, aWorld)
        self.assertTrue(np.allclose(a0Faces, [[1], [18.181818], [1818.1818], [181818.18]]))

        a0Faces = transport.harmonicMeanOverFaces(NWorldCoarse, NCoarseElement, 0, 1, aWorld)
        self.assertTrue(np.allclose(a0Faces, [[18.181818], [1818.1818], [181818.18], [1e7]]))
        
    def test_harmonicMeanOverFaces_2d(self):
        NWorldCoarse = np.array([3,4])
        NCoarseElement = np.array([4,2])
        NWorldFine = NWorldCoarse*NCoarseElement

        NtCoarse = np.prod(NWorldCoarse)
        NtFine = np.prod(NWorldFine)

        aWorld = np.ones(NtFine)

        aWorld[:NtFine//2] = 4
        aWorld[NtFine//2:] = 6
        
        aFaces = transport.harmonicMeanOverFaces(NWorldCoarse, NCoarseElement, 0, 0, aWorld)
        self.assertTrue(aFaces.shape == (NtCoarse, 2))
        self.assertTrue(np.allclose(aFaces[:NtCoarse//2], 4))
        self.assertTrue(np.allclose(aFaces[NtCoarse//2:], 6))

        aFaces = transport.harmonicMeanOverFaces(NWorldCoarse, NCoarseElement, 0, 1, aWorld)
        self.assertTrue(aFaces.shape == (NtCoarse, 2))
        self.assertTrue(np.allclose(aFaces[:NtCoarse//2], 4))
        self.assertTrue(np.allclose(aFaces[NtCoarse//2:], 6))

        aFaces = transport.harmonicMeanOverFaces(NWorldCoarse, NCoarseElement, 1, 0, aWorld)
        self.assertTrue(aFaces.shape == (NtCoarse, 4))
        self.assertTrue(np.allclose(aFaces, [[4]*4,   [4]*4,   [4]*4,
                                             [4]*4,   [4]*4,   [4]*4,
                                             [4.8]*4, [4.8]*4, [4.8]*4,
                                             [6]*4,   [6]*4,   [6]*4]))
        
        aFaces = transport.harmonicMeanOverFaces(NWorldCoarse, NCoarseElement, 1, 1, aWorld)
        self.assertTrue(aFaces.shape == (NtCoarse, 4))
        self.assertTrue(np.allclose(aFaces, [[4]*4,   [4]*4,   [4]*4,
                                             [4.8]*4, [4.8]*4, [4.8]*4,
                                             [6]*4,   [6]*4,   [6]*4,
                                             [6]*4,   [6]*4,   [6]*4]))

        aWorld[17] = 10
        aFaces = transport.harmonicMeanOverFaces(NWorldCoarse, NCoarseElement, 1, 1, aWorld)
        self.assertTrue(aFaces.shape == (NtCoarse, 4))
        self.assertTrue(np.allclose(aFaces, [[4]*4,   [4, 5.71428, 4, 4],   [4]*4,
                                             [4.8]*4, [4.8]*4, [4.8]*4,
                                             [6]*4,   [6]*4,   [6]*4,
                                             [6]*4,   [6]*4,   [6]*4]))
        
    def test_faceElementIndices_1d(self):
        NWorldCoarse = np.array([4])
        NCoarseElement = np.array([3])

        tIndices = transport.faceElementIndices(NWorldCoarse, NCoarseElement, 0, 0)
        self.assertTrue(np.all(tIndices == [[0], [3], [6], [9]]))

        tIndices = transport.faceElementIndices(NWorldCoarse, NCoarseElement, 0, 1)
        self.assertTrue(np.all(tIndices == [[2], [5], [8], [11]]))
        
    def test_faceElementIndices_2d(self):
        NWorldCoarse = np.array([3, 1])
        NCoarseElement = np.array([2, 3])

        tIndices = transport.faceElementIndices(NWorldCoarse, NCoarseElement, 0, 0)
        self.assertTrue(np.all(tIndices == [[0, 6, 12],
                                            [2, 8, 14],
                                            [4, 10, 16]]))

        tIndices = transport.faceElementIndices(NWorldCoarse, NCoarseElement, 0, 1)
        self.assertTrue(np.all(tIndices == [[1, 7, 13],
                                            [3, 9, 15],
                                            [5, 11, 17]]))

        tIndices = transport.faceElementIndices(NWorldCoarse, NCoarseElement, 1, 0)
        self.assertTrue(np.all(tIndices == [[0, 1],
                                            [2, 3],
                                            [4, 5]]))

        tIndices = transport.faceElementIndices(NWorldCoarse, NCoarseElement, 1, 1)
        self.assertTrue(np.all(tIndices == [[12, 13],
                                            [14, 15],
                                            [16, 17]]))

    def test_faceElementPointIndices_2d(self):
        NWorldCoarse = np.array([3, 1])
        NCoarseElement = np.array([2, 3])

        pIndices = transport.faceElementPointIndices(NWorldCoarse, NCoarseElement, 0, 0)
        self.assertTrue(np.all(pIndices == [[[0, 1, 7, 8],   [7, 8, 14, 15],   [14, 15, 21, 22]],
                                            [[2, 3, 9, 10],  [9, 10, 16, 17],  [16, 17, 23, 24]],
                                            [[4, 5, 11, 12], [11, 12, 18, 19], [18, 19, 25, 26]]]))
        
class computeHarmonicMeanFaceFlux_TestCase(unittest.TestCase):
    def test_computeHarmonicMeanFaceFlux_1d(self):
        NPatchCoarse = np.array([4])
        NCoarseElement = np.array([10])
        NPatchFine = NPatchCoarse*NCoarseElement

        NtFine = np.prod(NPatchFine)
        NpFine = np.prod(NPatchFine+1)

        aPatch = np.ones(NtFine)
        uPatch = np.ones(NpFine)
        
        fluxTF = transport.computeHarmonicMeanFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.all(fluxTF.shape == (4, 2)))
        self.assertTrue(np.all(fluxTF == 0))

        uPatch = np.linspace(0, 1, NpFine)
        
        fluxTF = transport.computeHarmonicMeanFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], 1) and np.allclose(fluxTF[:,1], -1))

        aPatch[0:10] = 5
        aPatch[10:20] = 100
        aPatch[20:30] = 3
        aPatch[30:40] = 2

        hm = lambda a, b: 2*a*b/float(a+b)
        
        fluxTF = transport.computeHarmonicMeanFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF, [[5,           -hm(5, 100)],
                                                 [hm(5, 100),  -hm(100, 3)],
                                                 [hm(100, 3),  -hm(3, 2)],
                                                 [hm(3, 2),    -2]]))

        aPatch[0] = 10
        aPatch[9] = 20
        aPatch[10] = 5
        aPatch[19] = 6
        fluxTF = transport.computeHarmonicMeanFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF, [[10,        -hm(20, 5)],
                                                 [hm(20, 5), -hm(6, 3)],
                                                 [hm(6, 3),  -hm(3, 2)],
                                                 [hm(3, 2),  -2]]))

        uPatch[10] = 1./40
        uPatch[9] = 2./40
        uPatch[11] = 0
        fluxTF = transport.computeHarmonicMeanFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF, [[10,          hm(20, 5)],
                                                 [-hm(20, 5), -hm(6, 3)],
                                                 [hm(6, 3),   -hm(3, 2)],
                                                 [hm(3, 2),   -2]]))
        
        fluxTF = transport.computeHarmonicMeanFaceFlux(10*NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF/10, [[10,          hm(20, 5)],
                                                    [-hm(20, 5), -hm(6, 3)],
                                                    [hm(6, 3),   -hm(3, 2)],
                                                    [hm(3, 2),   -2]]))
        
    def test_computeHarmonicMeanFaceFlux_2d(self):
        NPatchCoarse = np.array([4, 6])
        NCoarseElement = np.array([10, 20])
        NPatchFine = NPatchCoarse*NCoarseElement

        NtFine = np.prod(NPatchFine)
        NpFine = np.prod(NPatchFine+1)

        aPatch = np.ones(NtFine)
        uPatch = np.ones(NpFine)
        
        fluxTF = transport.computeHarmonicMeanFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.all(fluxTF.shape == (24, 4)))
        self.assertTrue(np.allclose(fluxTF, 0))

        x = util.pCoordinates(NPatchFine)[:,0]
        y = util.pCoordinates(NPatchFine)[:,1]
        uPatch = x

        xFaceArea = 1./NPatchCoarse[1]
        fluxTF = transport.computeHarmonicMeanFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], xFaceArea) and np.allclose(fluxTF[:,1], -xFaceArea))

        uPatch = x+y

        xFaceArea = 1./NPatchCoarse[1]
        yFaceArea = 1./NPatchCoarse[0]
        fluxTF = transport.computeHarmonicMeanFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], xFaceArea) and np.allclose(fluxTF[:,1], -xFaceArea))
        self.assertTrue(np.allclose(fluxTF[:,2], yFaceArea) and np.allclose(fluxTF[:,3], -yFaceArea))

        aPatch = 10*aPatch
        fluxTF = transport.computeHarmonicMeanFaceFlux(NPatchCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], 10*xFaceArea) and np.allclose(fluxTF[:,1], -10*xFaceArea))
        self.assertTrue(np.allclose(fluxTF[:,2], 10*yFaceArea) and np.allclose(fluxTF[:,3], -10*yFaceArea))

        aPatch = np.ones(NtFine)
        NWorldCoarse = np.array(NPatchCoarse)
        NWorldCoarse[0] = 10*NWorldCoarse[0]
        fluxTF = transport.computeHarmonicMeanFaceFlux(NWorldCoarse, NPatchCoarse, NCoarseElement, aPatch, uPatch)
        self.assertTrue(np.allclose(fluxTF[:,0], 10*xFaceArea) and np.allclose(fluxTF[:,1], -10*xFaceArea))
        self.assertTrue(np.allclose(fluxTF[:,2], 0.1*yFaceArea) and np.allclose(fluxTF[:,3], -0.1*yFaceArea))

class computeJumps_TestCase(unittest.TestCase):
    def test_computeJumps_1d(self):
        NWorldCoarse = np.array([4])
        quantityT = np.array([3., 6, 10, 2])
        
        jumpTF = transport.computeJumps(NWorldCoarse, quantityT)
        self.assertTrue(np.allclose(jumpTF, [[3, -3],
                                             [3, -4],
                                             [4, 8],
                                             [-8, 2]]))

    def test_computeJumps_2d(self):
        NWorldCoarse = np.array([2, 2])
        quantityT = np.array([3., 4, 5, 1])
        
        jumpTF = transport.computeJumps(NWorldCoarse, quantityT)
        self.assertTrue(np.allclose(jumpTF, [[3., -1, 3, -2],
                                             [1, 4, 4, 3],
                                             [5, 4, 2, 5],
                                             [-4, 1, -3, 1]]))

class computeConservativeFlux_TestCase(unittest.TestCase):
    def test_computeConservativeFlux_1d(self):
        NWorldCoarse = np.array([4])
        NCoarseElement = np.array([2])

        boundaryConditions = np.array([[0, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        fluxTF = np.array([[-1, 1],
                           [-1, 1],
                           [-1, 1],
                           [-1, 1]])
        conservativeFluxTF = transport.computeConservativeFlux(world, fluxTF)
        self.assertTrue(np.allclose(conservativeFluxTF, fluxTF))

        fluxTF = np.array([[-1, 2],
                           [-2, 1],
                           [-1, 1],
                           [-1, 1]])
        conservativeFluxTF = transport.computeConservativeFlux(world, fluxTF)
        self.assertTrue(np.allclose(np.sum(conservativeFluxTF, axis=1), 0))

        fluxTF = np.array([[-1, 2],
                           [-2, 3],
                           [-3, 2],
                           [-1, 1]])
        conservativeFluxTF = transport.computeConservativeFlux(world, fluxTF)
        self.assertTrue(np.allclose(np.sum(conservativeFluxTF, axis=1), 0))

    def test_computeConservativeFlux_2d_a(self):
        NWorldCoarse = np.array([2, 1])
        NCoarseElement = np.array([1, 1])

        boundaryConditions = np.array([[0, 0],
                                       [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        
        fluxTF = np.array([[-1., 2, 0, 0],
                           [-2, 2, 0, 0]])
        conservativeFluxTF = transport.computeConservativeFlux(world, fluxTF)
        self.assertTrue(np.allclose(np.sum(conservativeFluxTF, axis=1), 0))

    def test_computeConservativeFlux_2d_a(self):
        NWorldCoarse = np.array([2, 2])
        NCoarseElement = np.array([1, 1])

        boundaryConditions = np.array([[0, 0],
                                       [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        fluxTF = np.array([[-1., 2,  0, 0],
                           [-2 , 2,  0, 0],
                           [-2 , 3,  0, 0],
                           [-3 , 2,  0, 0]])
        conservativeFluxTF = transport.computeConservativeFlux(world, fluxTF)
        print conservativeFluxTF
        self.assertTrue(np.allclose(np.sum(conservativeFluxTF, axis=1), 0))
        
if __name__ == '__main__':
    #import cProfile
    #command = """unittest.main()"""
    #cProfile.runctx( command, globals(), locals(), filename="test_pg.profile" )
    unittest.main()
