import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

import fem
import util

class localMatrix_TestCase(unittest.TestCase):
    def test_massMatrix1d(self):
        M = fem.localMassMatrix(np.array([1]))
        self.assertTrue(np.allclose(M, 1./6*np.array([[2, 1], [1, 2]])))
        M = fem.localMassMatrix(np.array([10]))
        self.assertTrue(np.allclose(M, 1./10*1./6*np.array([[2, 1],
                                                         [1, 2]])))

    def test_massMatrix2d(self):
        M = fem.localMassMatrix(np.array([1,1]))
        self.assertTrue(np.allclose(M, 1./36*np.array([[4, 2, 2, 1],
                                                    [2, 4, 1, 2],
                                                    [2, 1, 4, 2],
                                                    [1, 2, 2, 4]])))
        M = fem.localMassMatrix(np.array([10,20]))
        self.assertTrue(np.allclose(M, 1./200*1./36*np.array([[4, 2, 2, 1],
                                                              [2, 4, 1, 2],
                                                              [2, 1, 4, 2],
                                                              [1, 2, 2, 4]])))

class stiffnessMatrix_TestCase(unittest.TestCase):
    def test_stiffnessMatrix1d(self):
        A = fem.localStiffnessMatrix(np.array([1]))
        self.assertTrue(np.allclose(A, np.array([[1, -1],
                                                 [-1, 1]])))
        A = fem.localStiffnessMatrix(np.array([10]))
        self.assertTrue(np.allclose(A, 10*np.array([[1, -1],
                                                    [-1, 1]])))

    def test_stiffnessMatrix2d(self):
        A = fem.localStiffnessMatrix(np.array([1,1]))
        self.assertTrue(np.allclose(A, 1./6*np.array([[4, -1, -1, -2],
                                                      [-1, 4, -2, -1],
                                                      [-1, -2, 4, -1],
                                                      [-2, -1, -1, 4]])))
        A = fem.localStiffnessMatrix(np.array([10,20]))
        self.assertTrue(np.allclose(A, 1./12*np.array([[10, 2, -7, -5],
                                                       [2, 10, -5, -7],
                                                       [-7, -5, 10, 2],
                                                       [-5, -7, 2, 10]])))
        
class boundaryNormalDerivativeMatrix_TestCase(unittest.TestCase):
    def test_boundaryNormalDerivativeMatrix1d(self):
        C = fem.localBoundaryNormalDerivativeMatrix(np.array([1]))
        self.assertTrue(np.allclose(C, np.array([[1, -1],
                                                 [0, 0]])))

        C = fem.localBoundaryNormalDerivativeMatrix(np.array([10]))
        self.assertTrue(np.allclose(C, 10.*np.array([[1, -1],
                                                     [0, 0]])))
        
    def test_boundaryNormalDerivativeMatrix2d(self):
        C = fem.localBoundaryNormalDerivativeMatrix(np.array([1, 1]))
        self.assertTrue(np.allclose(C, 1./6*np.array([[2, -2, 1, -1],
                                                      [0, 0, 0, 0],
                                                      [1, -1, 2, -2],
                                                      [0, 0, 0, 0]])))

        C = fem.localBoundaryNormalDerivativeMatrix(np.array([2, 10]))
        self.assertTrue(np.allclose(C, 2./10*1./6*np.array([[2, -2, 1, -1],
                                                            [0, 0, 0, 0],
                                                            [1, -1, 2, -2],
                                                            [0, 0, 0, 0]])))
        
        C = fem.localBoundaryNormalDerivativeMatrix(np.array([10, 2]))
        self.assertTrue(np.allclose(C, 10./2*1./6*np.array([[2, -2, 1, -1],
                                                            [0, 0, 0, 0],
                                                            [1, -1, 2, -2],
                                                            [0, 0, 0, 0]])))
        
        C = fem.localBoundaryNormalDerivativeMatrix(np.array([1, 1]), k=1)
        self.assertTrue(np.allclose(C, 1./6*np.array([[2, 1, -2, -1],
                                                      [1, 2, -1, -2],
                                                      [0, 0, 0, 0],
                                                      [0, 0, 0, 0]])))

        C = fem.localBoundaryNormalDerivativeMatrix(np.array([1, 1]), neg=True)
        self.assertTrue(np.allclose(C, 1./6*np.array([[0, 0, 0, 0],
                                                      [-2, 2, -1, 1],
                                                      [0, 0, 0, 0],
                                                      [-1, 1, -2, 2]])))

class boundaryMassMatrix_TestCase(unittest.TestCase):
    def test_boundaryMassMatrix1d(self):
        C = fem.localBoundaryMassMatrix(np.array([1]))
        self.assertTrue(np.allclose(C, np.array([[1, 0],
                                                 [0, 0]])))

        C = fem.localBoundaryMassMatrix(np.array([10]))
        self.assertTrue(np.allclose(C, np.array([[1, 0],
                                                 [0, 0]])))
        
    def test_boundaryMassMatrix2d(self):
        C = fem.localBoundaryMassMatrix(np.array([1, 1]))
        self.assertTrue(np.allclose(C, 1./6*np.array([[2, 0, 1, 0],
                                                      [0, 0, 0, 0],
                                                      [1, 0, 2, 0],
                                                      [0, 0, 0, 0]])))

        C = fem.localBoundaryMassMatrix(np.array([1, 2]))
        self.assertTrue(np.allclose(C, 1./2*1./6*np.array([[2, 0, 1, 0],
                                                           [0, 0, 0, 0],
                                                           [1, 0, 2, 0],
                                                           [0, 0, 0, 0]])))

        C = fem.localBoundaryMassMatrix(np.array([2, 1]))
        self.assertTrue(np.allclose(C, 1./6*np.array([[2, 0, 1, 0],
                                                      [0, 0, 0, 0],
                                                      [1, 0, 2, 0],
                                                      [0, 0, 0, 0]])))

        C = fem.localBoundaryMassMatrix(np.array([2, 1]), k=1)
        self.assertTrue(np.allclose(C, 1./2*1./6*np.array([[2, 1, 0, 0],
                                                           [1, 2, 0, 0],
                                                           [0, 0, 0, 0],
                                                           [0, 0, 0, 0]])))

        C = fem.localBoundaryMassMatrix(np.array([2, 1]), k=1, neg=True)
        self.assertTrue(np.allclose(C, 1./2*1./6*np.array([[0, 0, 0, 0],
                                                           [0, 0, 0, 0],
                                                           [0, 0, 2, 1],
                                                           [0, 0, 1, 2]])))
        
class assemblePatchMatrix_TestCase(unittest.TestCase):
    def test_assemblePatchMatrix2d(self):
        NPatch = np.array([2,2])
        ALoc = np.ones((4,4))
        AComputed = fem.assemblePatchMatrix(NPatch, ALoc)
        ACorrect = np.array([[1, 1, 0, 1, 1, 0, 0, 0, 0],
                             [1, 2, 1, 1, 2, 1, 0, 0, 0],
                             [0, 1, 1, 0, 1, 1, 0, 0, 0],
                             [1, 1, 0, 2, 2, 0, 1, 1, 0],
                             [1, 2, 1, 2, 4, 2, 1, 2, 1],
                             [0, 1, 1, 0, 2, 2, 0, 1, 1],
                             [0, 0, 0, 1, 1, 0, 1, 1, 0],
                             [0, 0, 0, 1, 2, 1, 1, 2, 1],
                             [0, 0, 0, 0, 1, 1, 0, 1, 1]])
        self.assertTrue(np.allclose(AComputed.todense(), ACorrect))

        NPatch = np.array([2,2])
        ALoc = np.ones((4,4))
        aPatch = [1, 2, 3, 4]
        AComputed = fem.assemblePatchMatrix(NPatch, ALoc, aPatch)
        ACorrect = np.array([[1, 1, 0, 1,  1, 0, 0, 0, 0],
                             [1, 3, 2, 1,  3, 2, 0, 0, 0],
                             [0, 2, 2, 0,  2, 2, 0, 0, 0],
                             [1, 1, 0, 4,  4, 0, 3, 3, 0],
                             [1, 3, 2, 4, 10, 6, 3, 7, 4],
                             [0, 2, 2, 0,  6, 6, 0, 4, 4],
                             [0, 0, 0, 3,  3, 0, 3, 3, 0],
                             [0, 0, 0, 3,  7, 4, 3, 7, 4],
                             [0, 0, 0, 0,  4, 4, 0, 4, 4]])
        self.assertTrue(np.allclose(AComputed.todense(), ACorrect))

class assemblePatchBoundaryMatrix_TestCase(unittest.TestCase):
    def test_assemblePatchBoundaryMatrix1d(self):
        NPatch = np.array([4])
        CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NPatch)
        CComputed = fem.assemblePatchBoundaryMatrix(NPatch, CLocGetter)
        CCorrect = np.array([[4, -4, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, -4, 4]])
        self.assertTrue(np.allclose(CComputed.todense(), CCorrect))

        NPatch = np.array([4])
        aPatch = np.array([2, 10, 10, 3])
        CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NPatch)
        CComputed = fem.assemblePatchBoundaryMatrix(NPatch, CLocGetter, aPatch)
        CCorrect = np.array([[8, -8, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0],
                             [0, 0, 0, -12, 12]])
        self.assertTrue(np.allclose(CComputed.todense(), CCorrect))

    def test_assemblePatchBoundaryNormalDerivativeMatrix2d(self):
        NPatch = np.array([1, 2])
        CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NPatch)
        CComputed = fem.assemblePatchBoundaryMatrix(NPatch, CLocGetter)
        CCorrect = 1./12*np.array([[10,  2,   1, -1,   0,  0],
                                   [ 2,  10, -1,  1,   0,  0],
                                   [-7, -5,   4, -4,  -7, -5],
                                   [-5, -7,  -4,  4,  -5, -7],
                                   [0, 0,     1, -1,   10, 2],
                                   [0, 0,    -1,  1,   2, 10]]).T
        self.assertTrue(np.allclose(CComputed.todense(), CCorrect))

        NPatch = np.array([1, 2])
        aPatch = np.array([1, 10])
        CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NPatch)
        CComputed = fem.assemblePatchBoundaryMatrix(NPatch, CLocGetter, aPatch)
        CCorrect = 1./12*np.array([[10,  2,   1,  -1,   0,   0],
                                   [ 2,  10, -1,   1,   0,   0],
                                   [-7, -5,  22, -22, -70, -50],
                                   [-5, -7, -22,  22, -50, -70],
                                   [0, 0,    10, -10, 100,  20],
                                   [0, 0,   -10,  10,  20, 100]]).T
        self.assertTrue(np.allclose(CComputed.todense(), CCorrect))

        
class stiffnessMatrixProperties_TestCase(unittest.TestCase):
    def test_stiffnessMatrixProperties(self):
        # Stiffness bilinear form should map constants to 0
        NPatch = np.array([3,4,5])
        ALoc = fem.localStiffnessMatrix(NPatch)
        aPatch = np.arange(np.prod(NPatch))
        A = fem.assemblePatchMatrix(NPatch, ALoc, aPatch)
        constant = np.ones(np.prod(NPatch+1))
        self.assertTrue(np.isclose(np.linalg.norm(A*constant), 0))

class massMatrixProperties_TestCase(unittest.TestCase):
    def test_massMatrixProperties(self):
        # Mass bilinear form should satisfy 1'*M*1 = 1
        NPatch = np.array([3,4,5])
        MLoc = fem.localMassMatrix(NPatch)
        M = fem.assemblePatchMatrix(NPatch, MLoc)
        ones = np.ones(np.prod(NPatch+1))
        self.assertTrue(np.isclose(np.linalg.norm(np.dot(ones, M*ones)), 1))

class boundaryNormalDerivativeMatrixProperties_TestCase(unittest.TestCase):
    def test_boundaryNormalDerivativeMatrixProperties(self):
        # BoundaryNormalDerivative bilinear form should map constants to 0
        NPatch = np.array([3,4,5])
        CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NPatch)
        C = fem.assemblePatchBoundaryMatrix(NPatch, CLocGetter)
        constant = np.ones(np.prod(NPatch+1))
        self.assertTrue(np.isclose(np.linalg.norm(C*constant), 0))

        # BoundaryNormalDerivative bilinear form (a=constant) should map planes to 0
        NPatch = np.array([3,4,5,6])
        CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NPatch)
        C = fem.assemblePatchBoundaryMatrix(NPatch, CLocGetter)

        p = util.pCoordinates(NPatch)
        pSum = np.sum(p, axis=1)
        ones = np.ones(np.prod(NPatch+1))
        self.assertTrue(np.isclose(np.linalg.norm(np.dot(ones, C*pSum)), 0))

        # A function f with df/dx_k = 1 at x_k = 0 and df/dx_k = 0 at
        # x_k = 1 should give 1'*C*f = -d
        NPatch = np.array([5,4,3,7])
        CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NPatch)
        C = fem.assemblePatchBoundaryMatrix(NPatch, CLocGetter)

        p = util.pCoordinates(NPatch)
        p = np.minimum(p, 0.5)
        pSum = np.sum(p, axis=1)
        ones = np.ones(np.prod(NPatch+1))
        self.assertTrue(np.isclose(np.dot(ones, C*pSum), -4))

        # Same test as above, but with a coefficient a
        NPatch = np.array([5,4,3,7])
        CLocGetter = fem.localBoundaryNormalDerivativeMatrixGetter(NPatch)

        p = util.pCoordinates(NPatch)
        p0 = p[:,0]
        pElement = util.pCoordinates(NPatch, NPatch=NPatch-1)
        pElement0 = pElement[:,0]
        aPatch = 1.*(pElement0<0.5) + 10.*(pElement0>=0.5)

        C = fem.assemblePatchBoundaryMatrix(NPatch, CLocGetter, aPatch)

        ones = np.ones(np.prod(NPatch+1))
        self.assertTrue(np.isclose(np.dot(ones, C*p0), 10-1))

class assembleProlongationMatrix_TestCase(unittest.TestCase):
    def test_assembleProlongationMatrixProperties(self):
        NPatchCoarse = np.array([2,2])
        NCoarseElement = np.array([4,4])

        P = fem.assembleProlongationMatrix(NPatchCoarse, NCoarseElement)
        self.assertTrue(np.isclose(np.linalg.norm(P.sum(axis=1)-1), 0))
        self.assertTrue(np.isclose(P[40,4], 1))

class assembleHierarchicalBasisMatrix_TestCase(unittest.TestCase):
    def test_assembleHierarchicalBasisProperties(self):
        NPatchCoarse = np.array([1])
        NCoarseElement = np.array([4])

        PHier = fem.assembleHierarchicalBasisMatrix(NPatchCoarse, NCoarseElement)
        self.assertTrue(np.allclose(PHier.A, np.array([[1,    0, 0,   0, 0   ],
                                                       [0.75, 1, 0.5, 0, 0.25],
                                                       [0.5,  0, 1,   0, 0.5 ],
                                                       [0.25, 0, 0.5, 1, 0.75],
                                                       [0.0,  0, 0,   0, 1   ]])))
        
        NPatchCoarse = np.array([1,1])
        NCoarseElement = np.array([4,4])
        NPatchFine = NPatchCoarse*NCoarseElement
        
        PHier = fem.assembleHierarchicalBasisMatrix(NPatchCoarse, NCoarseElement)
        self.assertTrue(np.allclose(PHier.diagonal(), 0*PHier.diagonal()+1))
        
if __name__ == '__main__':
    unittest.main()
