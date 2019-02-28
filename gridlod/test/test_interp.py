import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from gridlod import interp
from gridlod.world import World, Patch

class nodalPatchMatrix_TestCase(unittest.TestCase):
    def test_nodalPatchMatrix(self):
        NWorldCoarse = np.array([1, 1, 1])
        NCoarseElement = np.array([1, 1, 1])
        patch = Patch(World(NWorldCoarse, NCoarseElement), 1, 0)
        INodalPatch = interp.nodalPatchMatrix(patch)
        self.assertTrue(sparse.linalg.onenormest(INodalPatch - sparse.eye(np.size(INodalPatch,0))) == 0)

        NWorldCoarse = np.array([1, 1, 1])
        NCoarseElement = np.array([2, 1, 1])
        patch = Patch(World(NWorldCoarse, NCoarseElement), 1, 0)
        INodalPatch = interp.nodalPatchMatrix(patch)
        self.assertTrue(np.allclose(INodalPatch.todense(),
                                    np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])))

#class L2ProjectionPatchMatrix_TestCase(unittest.TestCase):
#    def test_L2ProjectionPatchMatrix(self):
#        iPatchCoarse = np.array([0, 0, 0])
#        NPatchCoarse = np.array([1, 1, 1])
#        NWorldCoarse = np.array([1, 1, 1])
#        NCoarseElement = np.array([1, 1, 1])
#        INodalPatch = interp.nodalPatchMatrix(iPatchCoarse, NPatchCoarse, NWorldCoarse, NCoarseElement)
#        self.assertTrue(sparse.linalg.onenormest(INodalPatch - sparse.eye(np.size(INodalPatch,0))) == 0)
        
if __name__ == '__main__':
    unittest.main()
