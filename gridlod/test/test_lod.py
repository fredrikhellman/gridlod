import unittest
import numpy as np
import itertools as it

from gridlod import lod, fem, interp, util, coef, world
from gridlod.world import World
from functools import reduce

class ritzProjectionToFinePatch_TestCase(unittest.TestCase):
    def test_trivial(self):
        NPatchCoarse = np.array([3,3])
        iPatchWorldCoarse = np.array([0, 0])
        NCoarseElement = np.array([2,2])
        NPatchFine = NPatchCoarse*NCoarseElement
        Nt = np.prod(NPatchFine)
        Np = np.prod(NPatchFine+1)
        fixed = util.boundarypIndexMap(NPatchFine)

        world = World(NPatchCoarse, NCoarseElement)
        
        aFlatPatchFine = np.ones(Nt)
        ALoc = fem.localStiffnessMatrix(NPatchFine)
        APatchFull = fem.assemblePatchMatrix(NPatchFine, ALoc, aFlatPatchFine)

        PPatch = fem.assembleProlongationMatrix(NPatchCoarse, NCoarseElement)

        IPatchNodal = interp.nodalPatchMatrix(np.array([0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)
        #IPatchuncL2 = interp.uncoupledL2ProjectionPatchMatrix(np.array([0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)
        IPatchL2 = interp.L2ProjectionPatchMatrix(np.array([0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)

        for IPatch in [IPatchNodal, IPatchL2]:
            np.random.seed(0)
            bPatchFullList = []
            self.assertTrue(not lod.ritzProjectionToFinePatch(world, iPatchWorldCoarse, NPatchCoarse,
                                                              APatchFull, bPatchFullList, IPatch))

            bPatchFullList = [np.zeros(Np)]
            projections = lod.ritzProjectionToFinePatch(world, iPatchWorldCoarse, NPatchCoarse,
                                                        APatchFull, bPatchFullList,
                                                        IPatch)
            self.assertEqual(len(projections), 1)
            self.assertTrue(np.allclose(projections[0], 0*projections[0]))

            bPatchFull = np.random.rand(Np)
            bPatchFullList = [bPatchFull]
            projections = lod.ritzProjectionToFinePatch(world, iPatchWorldCoarse, NPatchCoarse,
                                                        APatchFull, bPatchFullList,
                                                        IPatch)
            self.assertTrue(np.isclose(np.linalg.norm(IPatch*projections[0]), 0))
            
            self.assertTrue(np.isclose(np.dot(projections[0], APatchFull*projections[0]),
                                       np.dot(projections[0], bPatchFullList[0])))

            self.assertTrue(np.isclose(np.linalg.norm(projections[0][fixed]), 0))

            bPatchFullList = [bPatchFull, -bPatchFull]
            projections = lod.ritzProjectionToFinePatch(world, iPatchWorldCoarse, NPatchCoarse,
                                                        APatchFull, bPatchFullList,
                                                        IPatch)
            self.assertTrue(np.allclose(projections[0], -projections[1]))

            bPatchFullList = [np.random.rand(Np), np.random.rand(Np)]
            projections = lod.ritzProjectionToFinePatch(world, iPatchWorldCoarse, NPatchCoarse,
                                                        APatchFull, bPatchFullList,
                                                        IPatch)
            self.assertTrue(np.isclose(np.dot(projections[1], APatchFull*projections[0]),
                                       np.dot(projections[1], bPatchFullList[0])))

            bPatchFull = np.random.rand(Np)
            bPatchFullList = [bPatchFull]
            projectionCheckAgainst = lod.ritzProjectionToFinePatch(world, iPatchWorldCoarse, NPatchCoarse,
                                                                   APatchFull, bPatchFullList,
                                                                   IPatch)[0]

            for saddleSolver in [#lod.nullspaceOneLevelHierarchySolver(NPatchCoarse, NCoarseElement),
                                 lod.SchurComplementSolver()]:
                projection = lod.ritzProjectionToFinePatchWithGivenSaddleSolver(world, iPatchWorldCoarse, NPatchCoarse,                                                                             APatchFull, bPatchFullList,
                                                                                IPatch, saddleSolver)[0]
                self.assertTrue(np.isclose(np.max(np.abs(projectionCheckAgainst-projection)), 0))

            
        
class corrector_TestCase(unittest.TestCase):
    def test_init(self):
        NWorldCoarse = np.array([4, 4])
        NCoarseElement = np.array([2,2])
        world = World(NWorldCoarse, NCoarseElement)
        IPatchGenerator = lambda i, N: None

        k = 1

        iElementWorldCoarse = np.array([0, 0])
        ec = lod.ElementCorrector(world, k, iElementWorldCoarse, IPatchGenerator)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 0]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 0]))
        
        iElementWorldCoarse = np.array([0, 3])
        ec = lod.ElementCorrector(world, k, iElementWorldCoarse, IPatchGenerator)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 2]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 2]))

        iElementWorldCoarse = np.array([0, 2])
        ec = lod.ElementCorrector(world, k, iElementWorldCoarse, IPatchGenerator)
        self.assertTrue(np.all(ec.NPatchCoarse == [2, 3]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [0, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 1]))

        iElementWorldCoarse = np.array([1, 2])
        ec = lod.ElementCorrector(world, k, iElementWorldCoarse, IPatchGenerator)
        self.assertTrue(np.all(ec.NPatchCoarse == [3, 3]))
        self.assertTrue(np.all(ec.iElementPatchCoarse == [1, 1]))
        self.assertTrue(np.all(ec.iPatchWorldCoarse == [0, 1]))
        
    def test_testCsi(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        
        k = 1
        iElementWorldCoarse = np.array([2, 1, 2])
        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement)
        ec = lod.CoarseBasisElementCorrector(world, k, iElementWorldCoarse, IPatchGenerator)

        NtPatch = np.prod(ec.NPatchCoarse*NCoarseElement)

        np.random.seed(1)

        aPatch = np.random.rand(NtPatch)
        coefficientPatch = coef.CoefficientFine(ec.NPatchCoarse, NCoarseElement, aPatch)
        ec.computeCorrectors(coefficientPatch)
        ec.computeCoarseQuantities()

        TFinetIndexMap   = util.extractElementFine(ec.NPatchCoarse,
                                                   NCoarseElement,
                                                   ec.iElementPatchCoarse,
                                                   extractElements=True)
        TFinepIndexMap   = util.extractElementFine(ec.NPatchCoarse,
                                                   NCoarseElement,
                                                   ec.iElementPatchCoarse,
                                                   extractElements=False)
        TCoarsepIndexMap   = util.extractElementFine(ec.NPatchCoarse,
                                                     np.ones_like(NCoarseElement),
                                                     ec.iElementPatchCoarse,
                                                     extractElements=False)

        APatchFine      = fem.assemblePatchMatrix(ec.NPatchCoarse*NCoarseElement, world.ALocFine, aPatch)
        AElementFine    = fem.assemblePatchMatrix(NCoarseElement, world.ALocFine, aPatch[TFinetIndexMap])
        basisPatch      = fem.assembleProlongationMatrix(ec.NPatchCoarse, NCoarseElement)
        correctorsPatch = np.column_stack(ec.fsi.correctorsList)

        localBasis = world.localBasis

        KmsijShouldBe = -basisPatch.T*(APatchFine*(correctorsPatch))
        KmsijShouldBe[TCoarsepIndexMap,:] += np.dot(localBasis.T, AElementFine*localBasis)
        
        self.assertTrue(np.isclose(np.max(np.abs(ec.csi.Kmsij-KmsijShouldBe)), 0))
        
        
    def test_computeSingleT(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        
        k = 1
        iElementWorldCoarse = np.array([2, 1, 2])
        IPatchGenerator = lambda i, N: interp.nodalPatchMatrix(i, N, NWorldCoarse, NCoarseElement)
        ec = lod.CoarseBasisElementCorrector(world, k, iElementWorldCoarse, IPatchGenerator)

        NtPatch = np.prod(ec.NPatchCoarse*NCoarseElement)
        coefficientPatch = coef.CoefficientFine(ec.NPatchCoarse, NCoarseElement, np.ones(NtPatch))
        ec.computeCorrectors(coefficientPatch)

        correctorSum = reduce(np.add, ec.fsi.correctorsList)
        self.assertTrue(np.allclose(correctorSum, 0))

        ec.computeCoarseQuantities()
        # Test that the matrices have the constants in their null space
        #self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=1), 0))
        #self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=2), 0))

        self.assertTrue(np.allclose(np.sum(ec.csi.Kij, axis=0), 0))
        self.assertTrue(np.allclose(np.sum(ec.csi.Kij, axis=1), 0))
        self.assertTrue(np.allclose(np.sum(ec.csi.Kmsij, axis=0), 0))
        self.assertTrue(np.allclose(np.sum(ec.csi.Kmsij, axis=1), 0))

        # I had difficulties come up with test cases here. This test
        # verifies that most "energy" is in the element T.
        elementTIndex = util.convertpCoordIndexToLinearIndex(ec.NPatchCoarse-1, ec.iElementPatchCoarse)
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
        coefficientWorld = coef.CoefficientFine(NWorldCoarse, NCoarseElement, aWorld)
        k = np.max(NWorldCoarse)

        elementpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        elementpIndexMapFine = util.lowerLeftpIndexMap(NCoarseElement, NWorldFine)
        
        coarsepBasis = util.linearpIndexBasis(NWorldCoarse)
        finepBasis = util.linearpIndexBasis(NWorldFine)

        correctors = np.zeros((NpWorldFine, NpWorldCoarse))
        basis = np.zeros((NpWorldFine, NpWorldCoarse))
        
        for iElementWorldCoarse in it.product(*[np.arange(n, dtype='int64') for n in NWorldCoarse]):
            iElementWorldCoarse = np.array(iElementWorldCoarse)
            ec = lod.CoarseBasisElementCorrector(world, k, iElementWorldCoarse, lambda x, y: IWorld)
            ec.computeCorrectors(coefficientWorld)
            
            worldpIndices = np.dot(coarsepBasis, iElementWorldCoarse) + elementpIndexMap
            correctors[:,worldpIndices] += np.column_stack(ec.fsi.correctorsList)

            worldpFineIndices = np.dot(finepBasis, iElementWorldCoarse*NCoarseElement) + elementpIndexMapFine
            basis[np.ix_(worldpFineIndices, worldpIndices)] = world.localBasis

        AGlob = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aWorld)

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

    def test_ritzProjectionToFinePatchBoundaryConditions(self):
        NPatchCoarse = np.array([4, 4])
        iPatchWorldCoarse = np.array([0, 0])
        NCoarseElement = np.array([10, 10])
        world = World(NPatchCoarse, NCoarseElement)

        NPatchFine = NPatchCoarse*NCoarseElement
        NpFine = np.prod(NPatchFine + 1)
        
        APatchFull = fem.assemblePatchMatrix(NPatchCoarse*NCoarseElement, world.ALocFine)
        bPatchFullList = [np.ones(NpFine)]

        fixed = util.boundarypIndexMap(NPatchFine)
        
        for IPatch in [interp.L2ProjectionPatchMatrix(0*NPatchCoarse, NPatchCoarse, NPatchCoarse, NCoarseElement),
                       interp.nodalPatchMatrix(0*NPatchCoarse, NPatchCoarse, NPatchCoarse, NCoarseElement)]:

            schurComplementSolver = lod.SchurComplementSolver()
            schurComplementSolution = lod.ritzProjectionToFinePatchWithGivenSaddleSolver(world, iPatchWorldCoarse, NPatchCoarse,
                                                                                         APatchFull, bPatchFullList,
                                                                                         IPatch,
                                                                                         schurComplementSolver)[0]
            self.assertTrue(np.isclose(np.max(np.abs(schurComplementSolution[fixed])), 0))

if __name__ == '__main__':
    unittest.main()
