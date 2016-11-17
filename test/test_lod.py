import unittest
import numpy as np

import lod
import fem
import interp
import util

class computeElementCorrectorDirichletBC_TestCase(unittest.TestCase):
    def test_trivial(self):
        return
        NPatchCoarse = np.array([3,3])
        NCoarseElement = np.array([2,2])
        NPatchFine = NPatchCoarse*NCoarseElement
        iElementCoarse = np.array([1,1])
        aFlatPatchFine = np.ones(np.prod(NPatchFine))

        ALoc = fem.localStiffnessMatrix(NPatchFine)
        APatchFull = fem.assemblePatchMatrix(NPatchFine, ALoc, aFlatPatchFine)
        
        fineIndexBasis = util.linearpIndexBasis(NPatchFine)
        elementFineIndex = np.dot(fineIndexBasis, iElementCoarse*NCoarseElement)
        elementToFineIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine)
        AElementFull = fem.assemblePatchMatrix(NCoarseElement, ALoc,
                                               aFlatPatchFine[elementFineIndex + elementToFineIndexMap])
        
        localBasis = fem.localBasis(NCoarseElement)
        IPatch = interp.nodalPatchMatrix(np.array([0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)
        
        correctorsFull = lod.computeElementCorrectorDirichletBC(NPatchCoarse,
                                                                NCoarseElement,
                                                                iElementCoarse,
                                                                APatchFull,
                                                                AElementFull,
                                                                localBasis,
                                                                IPatch)

        # Test zeros at coarse points
        coarseIndices = util.fillpIndexMap(NPatchCoarse, NPatchFine)
        for correctorFull in correctorsFull:
            self.assertTrue(np.all(correctorFull[coarseIndices] == 0))

        # Test symmetry
        self.assertTrue(np.linalg.norm(correctorsFull[1].reshape([7,7]) -
                                       correctorsFull[0].reshape([7,7])[...,::-1]) < 1e-12)
        
        self.assertTrue(np.linalg.norm(correctorsFull[2].reshape([7,7]) -
                                       correctorsFull[0].reshape([7,7])[::-1,...]) < 1e-12)
        
        self.assertTrue(np.linalg.norm(correctorsFull[3].reshape([7,7]) -
                                       correctorsFull[0].reshape([7,7])[::-1,::-1]) < 1e-12)

        # They should sum to zero
        self.assertTrue(np.linalg.norm(reduce(np.add, correctorsFull)) < 1e-12)

    def test_writefile(self):
        return
        NPatchCoarse = np.array([5,5,5])
        NCoarseElement = np.array([3,3,3])
        NPatchFine = NPatchCoarse*NCoarseElement
        iElementCoarse = np.array([2,2,2])
        aFlatPatchFine = np.ones(np.prod(NPatchFine))
        aFlatPatchFine = np.exp(8*np.random.rand(np.prod(NPatchFine)))
        
        ALoc = fem.localStiffnessMatrix(NPatchFine)
        APatchFull = fem.assemblePatchMatrix(NPatchFine, ALoc, aFlatPatchFine)
        
        fineIndexBasis = util.linearpIndexBasis(NPatchFine)
        elementFineIndex = np.dot(fineIndexBasis, iElementCoarse*NCoarseElement)
        elementToFineIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine)
        AElementFull = fem.assemblePatchMatrix(NCoarseElement, ALoc,
                                               aFlatPatchFine[elementFineIndex + elementToFineIndexMap])
        
        localBasis = fem.localBasis(NCoarseElement)
        IPatch = interp.L2ProjectionPatchMatrix(np.array([0, 0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)
        
        correctorsFull = lod.computeElementCorrectorDirichletBC(NPatchCoarse,
                                                                NCoarseElement,
                                                                iElementCoarse,
                                                                APatchFull,
                                                                AElementFull,
                                                                localBasis,
                                                                IPatch)

        pointData = dict([('corrector_'+str(i), corrector.reshape(NPatchFine+1)) for i, corrector in enumerate(correctorsFull)])
        
        from pyevtk.hl import imageToVTK 
        imageToVTK("./correctors", pointData = pointData )
        
        
if __name__ == '__main__':
    unittest.main()
