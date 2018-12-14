import unittest
from gridlod import func, util

import numpy as np

class evaluate_TestCase(unittest.TestCase):
    def test_evaluateDQ0(self):
        N = np.array([3, 2])
        dq0 = np.array([10, 20, 30, 40, 50, 60])

        x = np.array([[0.1, 0.1],
                      [0.9, 0.1],
                      [0.5, 0.75],
                      [0.0, 0.0],
                      [1.0, 1.0]])

        dq0OfX = func.evaluateDQ0(N, dq0, x)
        dq0ShouldBe = np.array([10, 30, 50, 10, 60])

        self.assertTrue(np.allclose(dq0OfX, dq0ShouldBe))

    def test_evaluateDQ0TensorValued(self):
        N = np.array([3, 2])
        dq0 = np.array([[[10, 11]],
                        [[20, 21]],
                        [[30, 31]],
                        [[40, 41]],
                        [[50, 51]],
                        [[60, 61]]])

        x = np.array([[0.1, 0.1],
                      [0.9, 0.1],
                      [0.5, 0.75],
                      [0.0, 0.0],
                      [1.0, 1.0]])

        dq0OfX = func.evaluateDQ0(N, dq0, x)
        dq0ShouldBe = np.array([[[10, 11]],
                                [[30, 31]],
                                [[50, 51]],
                                [[10, 11]],
                                [[60, 61]]])

        self.assertTrue(np.allclose(dq0OfX, dq0ShouldBe))

    def test_evaluateCQ1_1d(self):
        N = np.array([3])
        cq1 = np.array([10, 20, 25, 0])

        x = np.array([[0],
                      [1./6],
                      [0.5],
                      [2./3],
                      [1.0]])

        cq1OfX = func.evaluateCQ1(N, cq1, x)
        cq1ShouldBe = np.array([10, 15, 22.5, 25, 0])

        self.assertTrue(np.allclose(cq1OfX, cq1ShouldBe))

    def test_evaluateCQ1_2d(self):
        N = np.array([3, 2])
        cq1 = np.array([10, 10, 30, 30,
                       60, 60, 80, 80,
                       60, 60, 90, 100])

        x = np.array([[0.1,  0.1],
                      [0.9,  0.1],
                      [0.5,  0.5],
                      [0.0,  0.0],
                      [1.0,  1.0],
                      [5./6, 1.0],
                      [5./6,  .75]])

        cq1OfX = func.evaluateCQ1(N, cq1, x)
        cq1ShouldBe = np.array([20, 40, 70, 10, 100, 95, 87.5])

        self.assertTrue(np.allclose(cq1OfX, cq1ShouldBe))
        

    def test_evaluateCQ1D_1d(self):
        N = np.array([3])
        cq1 = np.array([0.0, 1.0, 3.0, 103.0])

        x = np.array([[0.0],
                      [0.5],
                      [0.55],
                      [0.99]])

        cq1dOfX = func.evaluateCQ1D(N, cq1, x)
        print(cq1dOfX)
        cq1dShouldBe = np.array([[3.0], [6.0], [6.0], [300.0]])

        self.assertTrue(np.allclose(cq1dOfX, cq1dShouldBe))

    def test_evaluateCQ1D_2d(self):
        N = np.array([3, 2])
        cq1 = np.array([10, 10, 30, 30,
                        60, 60, 80, 80,
                        60, 60, 90, 100])

        x = np.array([[0.1, 0.1],
                      [0.5, 0.0],
                      [0.5, 0.75],
                      [0.1, 0.6]])

        cq1dOfX = func.evaluateCQ1D(N, cq1, x)
        cq1dShouldBe = np.array([[0.0,  100.0],
                                 [60.0, 100.0],
                                 [75.0,  10.0],
                                 [0.0,    0.0]])

        self.assertTrue(np.allclose(cq1dOfX, cq1dShouldBe))

    def test_evaluateCQ1D_2d_tensorValued(self):
        N = np.array([3, 2])
        cq1 = np.array([[10, 100], [10, 100], [30, 300], [30, 300],
                        [60, 600], [60, 600], [80, 800], [80, 800],
                        [60, 600], [60, 600], [90, 900], [100, 1000]])

        x = np.array([[0.1, 0.1],
                      [0.5, 0.0],
                      [0.5, 0.75],
                      [0.1, 0.6]])

        cq1dOfX = func.evaluateCQ1D(N, cq1, x)
        cq1dShouldBe = np.array([[[0.0,  100.0], [0,     1000.0]],
                                 [[60.0, 100.0], [600.0, 1000.0]],
                                 [[75.0,  10.0], [750.0,  100.0]],
                                 [[0.0,    0.0], [00.0,     0.0]]])

        self.assertTrue(np.allclose(cq1dOfX, cq1dShouldBe))
