import unittest
import numpy as np

from world import World
import lod
import fem
import interp
import util

class ritzProjectionToFinePatch_TestCase(unittest.TestCase):
    def test_trivial(self):
        NPatchCoarse = np.array([3,3])
        NCoarseElement = np.array([2,2])
        NPatchFine = NPatchCoarse*NCoarseElement
        Nt = np.prod(NPatchFine)
        Np = np.prod(NPatchFine+1)
        fixed = util.boundarypIndexMap(NPatchFine)

        aFlatPatchFine = np.ones(Nt)
        ALoc = fem.localStiffnessMatrix(NPatchFine)
        APatchFull = fem.assemblePatchMatrix(NPatchFine, ALoc, aFlatPatchFine)

        PPatch = fem.assembleProlongationMatrix(NPatchCoarse, NCoarseElement)

        IPatchNodal = interp.nodalPatchMatrix(np.array([0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)
        IPatchL2 = interp.uncoupledL2ProjectionPatchMatrix(np.array([0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)

        for IPatch in [IPatchNodal, IPatchL2]:
            np.random.seed(0)
            bPatchFullList = []
            self.assertTrue(not lod.ritzProjectionToFinePatch(NPatchCoarse,
                                                              NCoarseElement, APatchFull, bPatchFullList, IPatch))

            bPatchFullList = [np.zeros(Np)]
            projections = lod.ritzProjectionToFinePatch(NPatchCoarse, NCoarseElement,
                                                        APatchFull, bPatchFullList,
                                                        IPatch)
            self.assertEqual(len(projections), 1)
            self.assertTrue(np.allclose(projections[0], 0*projections[0]))

            bPatchFull = np.random.rand(Np)
            bPatchFullList = [bPatchFull]
            projections = lod.ritzProjectionToFinePatch(NPatchCoarse, NCoarseElement,
                                                        APatchFull, bPatchFullList,
                                                        IPatch)
            self.assertTrue(np.isclose(np.linalg.norm(IPatch*projections[0]), 0))
            self.assertTrue(np.isclose(np.dot(projections[0], APatchFull*projections[0]),
                                       np.dot(projections[0], bPatchFullList[0])))
            self.assertTrue(np.isclose(np.linalg.norm(projections[0][fixed]), 0))

            bPatchFullList = [bPatchFull, -bPatchFull]
            projections = lod.ritzProjectionToFinePatch(NPatchCoarse, NCoarseElement,
                                                        APatchFull, bPatchFullList,
                                                        IPatch)
            self.assertTrue(np.allclose(projections[0], -projections[1]))

            bPatchFullList = [np.random.rand(Np), np.random.rand(Np)]
            projections = lod.ritzProjectionToFinePatch(NPatchCoarse, NCoarseElement,
                                                        APatchFull, bPatchFullList,
                                                        IPatch)
            self.assertTrue(np.isclose(np.dot(projections[1], APatchFull*projections[0]),
                                       np.dot(projections[1], bPatchFullList[0])))
            
        
class corrector_TestCase(unittest.TestCase):
    def test_init(self):
        NWorldCoarse = np.array([4, 4])
        NCoarseElement = np.array([2,2])
        world = World(NWorldCoarse, NCoarseElement)

        k = 1

        iElementWorldCoarse = np.array([0, 0])
        ec = lod.ElementCorrector(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 0]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 0]))
        
        iElementWorldCoarse = np.array([0, 3])
        ec = lod.ElementCorrector(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 2]))

        iElementWorldCoarse = np.array([0, 2])
        ec = lod.ElementCorrector(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 3]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 1]))

        iElementWorldCoarse = np.array([1, 2])
        ec = lod.ElementCorrector(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [3, 3]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [1, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 1]))
        
    def test_compute(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)

        k = 1
        iElementWorldCoarse = np.array([1, 2, 0])
        ec = lod.ElementCorrector(world, k, iElementWorldCoarse)
        IPatch = interp.nodalPatchMatrix(ec.iPatchWorldCoarse, ec.NPatchCoarse, NWorldCoarse, NCoarseElement)

        NtPatch = np.prod(ec.NPatchCoarse*NCoarseElement)
        aPatch = np.ones(NtPatch)
        ec.computeCorrectors(aPatch, IPatch)

        correctorSum = reduce(np.add, ec.fsi.correctorsList)
        self.assertTrue(np.allclose(correctorSum, 0))

        ec.computeCoarseQuantities()
        self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=1), 0))
        self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=2), 0))

        self.assertTrue(np.allclose(np.sum(ec.csi.Kij, axis=0), 0))
        self.assertTrue(np.allclose(np.sum(ec.csi.Kij, axis=1), 0))

        ec.clearFineQuantities()
if __name__ == '__main__':
    unittest.main()




# class computeElementCorrectorDirichletBC_TestCase(unittest.TestCase):
#     def test_trivial(self):
#         return
#         NPatchCoarse = np.array([3,3])
#         NCoarseElement = np.array([2,2])
#         NPatchFine = NPatchCoarse*NCoarseElement
#         iElementCoarse = np.array([1,1])
#         aFlatPatchFine = np.ones(np.prod(NPatchFine))

#         ALoc = fem.localStiffnessMatrix(NPatchFine)
#         APatchFull = fem.assemblePatchMatrix(NPatchFine, ALoc, aFlatPatchFine)
        
#         fineIndexBasis = util.linearpIndexBasis(NPatchFine)
#         elementFineIndex = np.dot(fineIndexBasis, iElementCoarse*NCoarseElement)
#         elementToFineIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine)
#         AElementFull = fem.assemblePatchMatrix(NCoarseElement, ALoc,
#                                                aFlatPatchFine[elementFineIndex + elementToFineIndexMap])
        
#         localBasis = fem.localBasis(NCoarseElement)
#         IPatch = interp.nodalPatchMatrix(np.array([0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)
        
#         correctorsFull = lod.computeElementCorrectorDirichletBC(NPatchCoarse,
#                                                                 NCoarseElement,
#                                                                 iElementCoarse,
#                                                                 APatchFull,
#                                                                 AElementFull,
#                                                                 localBasis,
#                                                                 IPatch)

#         # Test zeros at coarse points
#         coarseIndices = util.fillpIndexMap(NPatchCoarse, NPatchFine)
#         for correctorFull in correctorsFull:
#             self.assertTrue(np.all(correctorFull[coarseIndices] == 0))

#         # Test symmetry
#         self.assertTrue(np.linalg.norm(correctorsFull[1].reshape([7,7]) -
#                                        correctorsFull[0].reshape([7,7])[...,::-1]) < 1e-12)
        
#         self.assertTrue(np.linalg.norm(correctorsFull[2].reshape([7,7]) -
#                                        correctorsFull[0].reshape([7,7])[::-1,...]) < 1e-12)
        
#         self.assertTrue(np.linalg.norm(correctorsFull[3].reshape([7,7]) -
#                                        correctorsFull[0].reshape([7,7])[::-1,::-1]) < 1e-12)

#         # They should sum to zero
#         self.assertTrue(np.linalg.norm(reduce(np.add, correctorsFull)) < 1e-12)

#     def test_writefile(self):
#         return
#         NPatchCoarse = np.array([5,5,5])
#         NCoarseElement = np.array([3,3,3])
#         NPatchFine = NPatchCoarse*NCoarseElement
#         iElementCoarse = np.array([2,2,2])
#         aFlatPatchFine = np.ones(np.prod(NPatchFine))
#         aFlatPatchFine = np.exp(8*np.random.rand(np.prod(NPatchFine)))
        
#         ALoc = fem.localStiffnessMatrix(NPatchFine)
#         APatchFull = fem.assemblePatchMatrix(NPatchFine, ALoc, aFlatPatchFine)
        
#         fineIndexBasis = util.linearpIndexBasis(NPatchFine)
#         elementFineIndex = np.dot(fineIndexBasis, iElementCoarse*NCoarseElement)
#         elementToFineIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine)
#         AElementFull = fem.assemblePatchMatrix(NCoarseElement, ALoc,
#                                                aFlatPatchFine[elementFineIndex + elementToFineIndexMap])
        
#         localBasis = fem.localBasis(NCoarseElement)
#         IPatch = interp.L2ProjectionPatchMatrix(np.array([0, 0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)
        
#         correctorsFull = lod.computeElementCorrectorDirichletBC(NPatchCoarse,
#                                                                 NCoarseElement,
#                                                                 iElementCoarse,
#                                                                 APatchFull,
#                                                                 AElementFull,
#                                                                 localBasis,
#                                                                 IPatch)

#         pointData = dict([('corrector_'+str(i), corrector.reshape(NPatchFine+1)) for i, corrector in enumerate(correctorsFull)])
        
#         from pyevtk.hl import imageToVTK 
#         imageToVTK("./correctors", pointData = pointData )
        
        
    
