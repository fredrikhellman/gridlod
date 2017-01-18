import unittest
import numpy as np
import itertools as it

from world import World
import lod
import fem
import interp
import util
import coef

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
        ec = lod.elementCorrector(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 0]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 0]))
        
        iElementWorldCoarse = np.array([0, 3])
        ec = lod.elementCorrector(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 2]))

        iElementWorldCoarse = np.array([0, 2])
        ec = lod.elementCorrector(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 3]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 1]))

        iElementWorldCoarse = np.array([1, 2])
        ec = lod.elementCorrector(world, k, iElementWorldCoarse)
        self.assertTrue(np.all(ec.NPatchCoarse == [3, 3]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [1, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 1]))
        
    def test_computeSingleT(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        
        k = 1
        iElementWorldCoarse = np.array([2, 1, 2])
        ec = lod.elementCorrector(world, k, iElementWorldCoarse)
        IPatch = interp.nodalPatchMatrix(ec.iPatchWorldCoarse, ec.NPatchCoarse, NWorldCoarse, NCoarseElement)

        NtPatch = np.prod(ec.NPatchCoarse*NCoarseElement)
        coefficientPatch = coef.coefficientFine(np.ones(NtPatch))
        ec.computeCorrectors(coefficientPatch, IPatch)

        correctorSum = reduce(np.add, ec.fsi.correctorsList)
        self.assertTrue(np.allclose(correctorSum, 0))

        ec.computeCoarseQuantities()
        # Test that the matrices have the constants in their null space
        #self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=1), 0))
        #self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=2), 0))

        self.assertTrue(np.allclose(np.sum(ec.csi.Kij, axis=0), 0))
        self.assertTrue(np.allclose(np.sum(ec.csi.Kij, axis=1), 0))

        # I had difficulties come up with test cases here. This test
        # verifies that most "energy" is in the element T.
        elementTIndex = util.convertpCoordinateToIndex(ec.NPatchCoarse-1, ec.iElementPatchCoarse)
        self.assertTrue(np.all(ec.csi.muTPrime[elementTIndex] >= ec.csi.muTPrime))
        self.assertTrue(not np.all(ec.csi.muTPrime[elementTIndex+1] >= ec.csi.muTPrime))
        ec.clearFineQuantities()

    def test_computeFullDomain(self):
        NWorldCoarse = np.array([2, 3, 4], dtype='int64')
        NWorldCoarse = np.array([1, 1, 1], dtype='int64')
        NCoarseElement = np.array([4, 2, 3], dtype='int64')
        NWorldFine = NWorldCoarse*NCoarseElement
        NpWorldFine = np.prod(NWorldFine+1)
        NpWorldCoarse = np.prod(NWorldCoarse+1)
        NtWorldFine = np.prod(NWorldCoarse*NCoarseElement)

        np.random.seed(0)

        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        IWorld = interp.nodalPatchMatrix(0*NWorldCoarse, NWorldCoarse, NWorldCoarse, NCoarseElement)
        aWorld = np.exp(np.random.rand(NtWorldFine))
        coefficientWorld = coef.coefficientFine(aWorld)
        k = np.max(NWorldCoarse)

        elementpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        elementpIndexMapFine = util.lowerLeftpIndexMap(NCoarseElement, NWorldFine)
        
        coarsepBasis = util.linearpIndexBasis(NWorldCoarse)
        finepBasis = util.linearpIndexBasis(NWorldFine)

        correctors = np.zeros((NpWorldFine, NpWorldCoarse))
        basis = np.zeros((NpWorldFine, NpWorldCoarse))
        
        for iElementWorldCoarse in it.product(*[np.arange(n, dtype='int64') for n in NWorldCoarse]):
            iElementWorldCoarse = np.array(iElementWorldCoarse)
            ec = lod.elementCorrector(world, k, iElementWorldCoarse)
            ec.computeCorrectors(coefficientWorld, IWorld)
            
            worldpIndices = np.dot(coarsepBasis, iElementWorldCoarse) + elementpIndexMap
            correctors[:,worldpIndices] += np.column_stack(ec.fsi.correctorsList)

            worldpFineIndices = np.dot(finepBasis, iElementWorldCoarse*NCoarseElement) + elementpIndexMapFine
            basis[np.ix_(worldpFineIndices, worldpIndices)] = world.localBasis

        AGlob = fem.assemblePatchMatrix(NWorldFine, world.ALoc, aWorld)

        alpha = np.random.rand(NpWorldCoarse)
        vH  = np.dot(basis, alpha)
        QvH = np.dot(correctors, alpha)

        # Check norm inequality
        self.assertTrue(np.dot(QvH.T, AGlob*QvH) <= np.dot(vH.T, AGlob*vH))

        # Check that correctors are really fine functions
        self.assertTrue(np.isclose(np.linalg.norm(IWorld*correctors, ord=np.inf), 0))

        v = np.random.rand(NpWorldFine, NpWorldCoarse)
        v[util.boundarypIndexMap(NWorldFine)] = 0
        # The chosen interpolation operator doesn't ruin the boundary conditions.
        vf = v-np.dot(basis, IWorld*v)
        vf = vf/np.sqrt(np.sum(vf*(AGlob*vf), axis=0))
        # Check orthogonality
        self.assertTrue(np.isclose(np.linalg.norm(np.dot(vf.T, AGlob*(correctors - basis)), ord=np.inf), 0))

    def test_computeErrorIndicator(self):
        NWorldCoarse = np.array([7, 7], dtype='int64')
        NCoarseElement = np.array([10, 10], dtype='int64')
        NWorldFine = NWorldCoarse*NCoarseElement
        NpWorldFine = np.prod(NWorldFine+1)
        NpWorldCoarse = np.prod(NWorldCoarse+1)
        NtWorldFine = np.prod(NWorldCoarse*NCoarseElement)
        NtWorldCoarse = np.prod(NWorldCoarse)

        np.random.seed(0)

        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        aBase = np.exp(np.random.rand(NtWorldFine))
        k = np.max(NWorldCoarse)
        iElementWorldCoarse = np.array([3,3])

        rCoarseFirst = 1+3*np.random.rand(NtWorldCoarse)
        coefFirst = coef.coefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarseFirst)
        ec = lod.elementCorrector(world, k, iElementWorldCoarse)
        IPatch = interp.L2ProjectionPatchMatrix(ec.iPatchWorldCoarse, ec.NPatchCoarse, NWorldCoarse, NCoarseElement)
        ec.computeCorrectors(coefFirst, IPatch)
        ec.computeCoarseQuantities()

        # If both rCoarseFirst and rCoarseSecond are equal, the error indicator should be zero
        rCoarseSecond = np.array(rCoarseFirst)
        self.assertTrue(np.isclose(ec.computeErrorIndicator(rCoarseSecond), 0))

        # If rCoarseSecond is not rCoarseFirst, the error indicator should not be zero
        rCoarseSecond = 2*np.array(rCoarseFirst)
        self.assertTrue(ec.computeErrorIndicator(rCoarseSecond) >= 0.1)

        # If rCoarseSecond is different in the element itself, the error
        # indicator should be large
        elementCoarseIndex = util.convertpCoordinateToIndex(NWorldCoarse-1, iElementWorldCoarse)
        rCoarseSecond = np.array(rCoarseFirst)
        rCoarseSecond[elementCoarseIndex] *= 2
        saveForNextTest = ec.computeErrorIndicator(rCoarseSecond)
        self.assertTrue(saveForNextTest >= 0.1)

        # A difference in the perifery should be smaller than in the center
        rCoarseSecond = np.array(rCoarseFirst)
        rCoarseSecond[0] *= 2
        self.assertTrue(saveForNextTest > ec.computeErrorIndicator(rCoarseSecond))

        # Again, but closer
        rCoarseSecond = np.array(rCoarseFirst)
        rCoarseSecond[elementCoarseIndex-1] *= 2
        self.assertTrue(saveForNextTest > ec.computeErrorIndicator(rCoarseSecond))
        
        
if __name__ == '__main__':
    #    import cProfile
    #    command = """unittest.main()"""
    #    cProfile.runctx( command, globals(), locals(), filename="test_lod.profile" )
    unittest.main()




# class computeelementCorrectorDirichletBC_TestCase(unittest.TestCase):
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
        
#         correctorsFull = lod.computeelementCorrectorDirichletBC(NPatchCoarse,
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
        
#         correctorsFull = lod.computeelementCorrectorDirichletBC(NPatchCoarse,
#                                                                 NCoarseElement,
#                                                                 iElementCoarse,
#                                                                 APatchFull,
#                                                                 AElementFull,
#                                                                 localBasis,
#                                                                 IPatch)

#         pointData = dict([('corrector_'+str(i), corrector.reshape(NPatchFine+1)) for i, corrector in enumerate(correctorsFull)])
        
#         from pyevtk.hl import imageToVTK 
#         imageToVTK("./correctors", pointData = pointData )
        
        
    
