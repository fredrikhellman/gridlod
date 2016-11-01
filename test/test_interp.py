import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

import interp

class nodalPatchMatrix_TestCase(unittest.TestCase):
    def runTest(self):
        iPatchCoarse = np.array([0, 0, 0])
        NPatchCoarse = np.array([1, 1, 1])
        NWorldCoarse = np.array([1, 1, 1])
        NCoarseElement = np.array([1, 1, 1])
        INodalPatch = interp.nodalPatchMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse, NCoarseElement)
        self.assertTrue(sparse.linalg.onenormest(INodalPatch - sparse.eye(np.size(INodalPatch,0))) == 0)
       
if __name__ == '__main__':
    unittest.main()
