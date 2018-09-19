import unittest
from gridlod import func, util

import numpy as np

class evaluate_TestCase(unittest.TestCase):
    def test_evaluateP0(self):
        N = np.array([3, 2])
        p0 = np.array([10, 20, 30, 40, 50, 60])

        x = np.array([[0.1, 0.1],
                      [0.9, 0.1],
                      [0.5, 0.75],
                      [0.0, 0.0],
                      [1.0, 1.0]])

        p0OfX = func.evaluateP0(N, p0, x)
        p0ShouldBe = np.array([10, 30, 50, 10, 60])

        self.assertTrue(np.allclose(p0OfX, p0ShouldBe))

    def test_evaluateP0TensorValued(self):
        N = np.array([3, 2])
        p0 = np.array([[[10, 11]],
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

        p0OfX = func.evaluateP0(N, p0, x)
        p0ShouldBe = np.array([[[10, 11]],
                               [[30, 31]],
                               [[50, 51]],
                               [[10, 11]],
                               [[60, 61]]])

        self.assertTrue(np.allclose(p0OfX, p0ShouldBe))

    def test_evaluateP1(self):
        N = np.array([3, 2])
        p1 = np.array([10, 10, 30, 30,
                       60, 60, 80, 80,
                       60, 60, 90, 100])

        x = np.array([[0.1,  0.1],
                      [0.9,  0.1],
                      [0.5,  0.5],
                      [0.0,  0.0],
                      [1.0,  1.0],
                      [5./6, 1.0],
                      [5./6,  .75]])

        p1OfX = func.evaluateP1(N, p1, x)
        p1ShouldBe = np.array([20, 40, 70, 10, 100, 95, 87.5])

        self.assertTrue(np.allclose(p1OfX, p1ShouldBe))
        
