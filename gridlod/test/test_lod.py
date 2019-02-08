import unittest
import numpy as np
import itertools as it

from gridlod import lod, fem, interp, util, coef, world
from gridlod.world import World, Patch
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
                projection = lod.ritzProjectionToFinePatch(world, iPatchWorldCoarse, NPatchCoarse,
                                                           APatchFull, bPatchFullList,
                                                           IPatch, saddleSolver)[0]
                self.assertTrue(np.isclose(np.max(np.abs(projectionCheckAgainst-projection)), 0))

            
        
class corrector_TestCase(unittest.TestCase):
    def test_testCsi(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        
        k = 1
        iElementWorldCoarse = np.array([2, 1, 2])
        TInd = util.convertpCoordIndexToLinearIndex(NWorldCoarse, iElementWorldCoarse)
        patch = Patch(world, k, TInd)
        
        IPatch = interp.L2ProjectionPatchMatrix(patch.iPatchWorldCoarse, patch.NPatchCoarse, NWorldCoarse, NCoarseElement)
        
        NtPatch = patch.NtFine
        np.random.seed(1)
        aPatch = np.random.rand(NtPatch)
        basisCorrectorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeCoarseQuantities(patch, basisCorrectorsList, aPatch)

        TFinetIndexMap   = util.extractElementFine(patch.NPatchCoarse,
                                                   NCoarseElement,
                                                   patch.iElementPatchCoarse,
                                                   extractElements=True)
        TFinepIndexMap   = util.extractElementFine(patch.NPatchCoarse,
                                                   NCoarseElement,
                                                   patch.iElementPatchCoarse,
                                                   extractElements=False)
        TCoarsepIndexMap   = util.extractElementFine(patch.NPatchCoarse,
                                                     np.ones_like(NCoarseElement),
                                                     patch.iElementPatchCoarse,
                                                     extractElements=False)

        APatchFine      = fem.assemblePatchMatrix(patch.NPatchFine, world.ALocFine, aPatch)
        AElementFine    = fem.assemblePatchMatrix(NCoarseElement, world.ALocFine, aPatch[TFinetIndexMap])
        basisPatch      = fem.assembleProlongationMatrix(patch.NPatchCoarse, NCoarseElement)
        correctorsPatch = np.column_stack(basisCorrectorsList)

        localBasis = world.localBasis

        KmsijShouldBe = -basisPatch.T*(APatchFine*(correctorsPatch))
        KmsijShouldBe[TCoarsepIndexMap,:] += np.dot(localBasis.T, AElementFine*localBasis)
        
        self.assertTrue(np.isclose(np.max(np.abs(csi.Kmsij-KmsijShouldBe)), 0))
        
        
    def test_computeSingleT(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        
        k = 1
        iElementWorldCoarse = np.array([2, 1, 2])
        TInd = util.convertpCoordIndexToLinearIndex(NWorldCoarse, iElementWorldCoarse)
        patch = Patch(world, k, TInd)

        IPatch = interp.L2ProjectionPatchMatrix(patch.iPatchWorldCoarse, patch.NPatchCoarse, NWorldCoarse, NCoarseElement)
        
        NtPatch = patch.NtFine

        aPatch = np.ones(NtPatch)
        basisCorrectorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)

        correctorSum = reduce(np.add, basisCorrectorsList)
        self.assertTrue(np.allclose(correctorSum, 0))

        csi = lod.computeCoarseQuantities(patch, basisCorrectorsList, aPatch)
        # Test that the matrices have the constants in their null space
        #self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=1), 0))
        #self.assertTrue(np.allclose(np.sum(ec.csi.LTPrimeij, axis=2), 0))

        self.assertTrue(np.allclose(np.sum(csi.Kij, axis=0), 0))
        self.assertTrue(np.allclose(np.sum(csi.Kij, axis=1), 0))
        self.assertTrue(np.allclose(np.sum(csi.Kmsij, axis=0), 0))
        self.assertTrue(np.allclose(np.sum(csi.Kmsij, axis=1), 0))

        # I had difficulties come up with test cases here. This test
        # verifies that most "energy" is in the element T.
        elementTIndex = util.convertpCoordIndexToLinearIndex(patch.NPatchCoarse-1, patch.iElementPatchCoarse)
        self.assertTrue(np.all(csi.muTPrime[elementTIndex] >= csi.muTPrime))
        self.assertTrue(not np.all(csi.muTPrime[elementTIndex+1] >= csi.muTPrime))

    def test_computeFullDomain(self):
        NWorldCoarse = np.array([1, 1, 1], dtype='int64')
        NCoarseElement = np.array([4, 2, 3], dtype='int64')
        NWorldFine = NWorldCoarse*NCoarseElement

        np.random.seed(0)

        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        IWorld = interp.nodalPatchMatrix(0*NWorldCoarse, NWorldCoarse, NWorldCoarse, NCoarseElement)
        aWorld = np.exp(np.random.rand(world.NtFine))
        k = np.max(NWorldCoarse)

        elementpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        elementpIndexMapFine = util.lowerLeftpIndexMap(NCoarseElement, NWorldFine)
        
        coarsepBasis = util.linearpIndexBasis(NWorldCoarse)
        finepBasis = util.linearpIndexBasis(NWorldFine)

        correctors = np.zeros((world.NpFine, world.NpCoarse))
        basis = np.zeros((world.NpFine, world.NpCoarse))
        
        for iElementWorldCoarse in it.product(*[np.arange(n, dtype='int64') for n in NWorldCoarse]):
            iElementWorldCoarse = np.array(iElementWorldCoarse)
            TInd = util.convertpCoordIndexToLinearIndex(NWorldCoarse, iElementWorldCoarse)
            patch = Patch(world, k, TInd)
            
            correctorsList = lod.computeBasisCorrectors(patch, IWorld, aWorld)
            
            worldpIndices = np.dot(coarsepBasis, iElementWorldCoarse) + elementpIndexMap
            correctors[:,worldpIndices] += np.column_stack(correctorsList)

            worldpFineIndices = np.dot(finepBasis, iElementWorldCoarse*NCoarseElement) + elementpIndexMapFine
            basis[np.ix_(worldpFineIndices, worldpIndices)] = world.localBasis

        AGlob = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aWorld)

        alpha = np.random.rand(world.NpCoarse)
        vH  = np.dot(basis, alpha)
        QvH = np.dot(correctors, alpha)

        # Check norm inequality
        self.assertTrue(np.dot(QvH.T, AGlob*QvH) <= np.dot(vH.T, AGlob*vH))

        # Check that correctors are really fine functions
        self.assertTrue(np.isclose(np.linalg.norm(IWorld*correctors, ord=np.inf), 0))

        v = np.random.rand(world.NpFine, world.NpCoarse)
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
            schurComplementSolution = lod.ritzProjectionToFinePatch(world, iPatchWorldCoarse, NPatchCoarse,
                                                                    APatchFull, bPatchFullList,
                                                                    IPatch,
                                                                    schurComplementSolver)[0]
            self.assertTrue(np.isclose(np.max(np.abs(schurComplementSolution[fixed])), 0))

if __name__ == '__main__':
    unittest.main()
