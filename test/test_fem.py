import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

import fem

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
class boundaryMatrix_TestCase(unittest.TestCase):
    def test_boundaryMatrix1d(self):
        C = fem.localBoundaryMatrix(np.array([1]))
        self.assertTrue(np.allclose(C, np.array([[1, -1],
                                                 [0, 0]])))

        C = fem.localBoundaryMatrix(np.array([10]))
        self.assertTrue(np.allclose(C, 10.*np.array([[1, -1],
                                                     [0, 0]])))
        
    def test_boundaryMatrix2d(self):
        C = fem.localBoundaryMatrix(np.array([1, 1]))
        self.assertTrue(np.allclose(C, 1./6*np.array([[2, 1, -2, -1],
                                                      [1, 2, -1, -2],
                                                      [0, 0, 0, 0],
                                                      [0, 0, 0, 0]])))

        C = fem.localBoundaryMatrix(np.array([2, 10]))
        self.assertTrue(np.allclose(C, 1./2*10*1./6*np.array([[2, 1, -2, -1],
                                                              [1, 2, -1, -2],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0]])))
        
        C = fem.localBoundaryMatrix(np.array([10, 2]))
        self.assertTrue(np.allclose(C, 1./10*2*1./6*np.array([[2, 1, -2, -1],
                                                              [1, 2, -1, -2],
                                                              [0, 0, 0, 0],
                                                              [0, 0, 0, 0]])))

if __name__ == '__main__':
    unittest.main()
