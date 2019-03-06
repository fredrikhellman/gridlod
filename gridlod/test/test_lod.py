import unittest
import numpy as np
import itertools as it

from gridlod import lod, fem, interp, util, coef, world
from gridlod.world import World, Patch
from functools import reduce

class ritzProjectionToFinePatch_TestCase(unittest.TestCase):
    def test_trivial(self):
        NPatchCoarse = np.array([3,3])
        NCoarseElement = np.array([2,2])
        NPatchFine = NPatchCoarse*NCoarseElement
        Nt = np.prod(NPatchFine)
        Np = np.prod(NPatchFine+1)
        fixed = util.boundarypIndexMap(NPatchFine)

        world = World(NPatchCoarse, NCoarseElement)
        patch = Patch(world, 3, 0)
        
        aFlatPatchFine = np.ones(Nt)
        ALoc = fem.localStiffnessMatrix(NPatchFine)
        APatchFull = fem.assemblePatchMatrix(NPatchFine, ALoc, aFlatPatchFine)

        PPatch = fem.assembleProlongationMatrix(NPatchCoarse, NCoarseElement)

        IPatchNodal = interp.nodalPatchMatrix(patch)
        #IPatchuncL2 = interp.uncoupledL2ProjectionPatchMatrix(np.array([0, 0]), NPatchCoarse, NPatchCoarse, NCoarseElement)
        IPatchL2 = interp.L2ProjectionPatchMatrix(patch)

        for IPatch in [IPatchNodal, IPatchL2]:
            np.random.seed(0)
            bPatchFullList = []
            self.assertTrue(not lod.ritzProjectionToFinePatch(patch, APatchFull, bPatchFullList, IPatch))

            bPatchFullList = [np.zeros(Np)]
            projections = lod.ritzProjectionToFinePatch(patch, APatchFull, bPatchFullList, IPatch)
            self.assertEqual(len(projections), 1)
            self.assertTrue(np.allclose(projections[0], 0*projections[0]))

            bPatchFull = np.random.rand(Np)
            bPatchFullList = [bPatchFull]
            projections = lod.ritzProjectionToFinePatch(patch, APatchFull, bPatchFullList, IPatch)
            self.assertTrue(np.isclose(np.linalg.norm(IPatch*projections[0]), 0))
            
            self.assertTrue(np.isclose(np.dot(projections[0], APatchFull*projections[0]),
                                       np.dot(projections[0], bPatchFullList[0])))

            self.assertTrue(np.isclose(np.linalg.norm(projections[0][fixed]), 0))

            bPatchFullList = [bPatchFull, -bPatchFull]
            projections = lod.ritzProjectionToFinePatch(patch, APatchFull, bPatchFullList, IPatch)
            self.assertTrue(np.allclose(projections[0], -projections[1]))

            bPatchFullList = [np.random.rand(Np), np.random.rand(Np)]
            projections = lod.ritzProjectionToFinePatch(patch, APatchFull, bPatchFullList, IPatch)
            self.assertTrue(np.isclose(np.dot(projections[1], APatchFull*projections[0]),
                                       np.dot(projections[1], bPatchFullList[0])))

            bPatchFull = np.random.rand(Np)
            bPatchFullList = [bPatchFull]
            projectionCheckAgainst = lod.ritzProjectionToFinePatch(patch, APatchFull, bPatchFullList, IPatch)[0]

            for saddleSolver in [#lod.nullspaceOneLevelHierarchySolver(NPatchCoarse, NCoarseElement),
                                 lod.SchurComplementSolver()]:
                projection = lod.ritzProjectionToFinePatch(patch, APatchFull, bPatchFullList,
                                                           IPatch, saddleSolver)[0]
                self.assertTrue(np.isclose(np.max(np.abs(projectionCheckAgainst-projection)), 0))

class corrector_TestCase(unittest.TestCase):
    def test_testCsi_Kmsij(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        
        k = 1
        iElementWorldCoarse = np.array([2, 1, 2])
        TInd = util.convertpCoordIndexToLinearIndex(NWorldCoarse, iElementWorldCoarse)
        patch = Patch(world, k, TInd)
        
        IPatch = interp.L2ProjectionPatchMatrix(patch)
        
        NtPatch = patch.NtFine
        np.random.seed(1)
        aPatch = np.random.rand(NtPatch)
        basisCorrectorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
        csi = lod.computeBasisCoarseQuantities(patch, basisCorrectorsList, aPatch)

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

    def test_testCsi_muTPrime(self):
        # 3D world
        NWorldCoarse = np.array([6, 5, 4])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)

        # Full patch
        TInd = 0
        k = 6
        patch = Patch(world, k, TInd)

        # Let functions = [x1]
        def computeFunctions():
            pc = util.pCoordinates(world.NWorldFine)
            x1 = pc[:,0]
            x2 = pc[:,1]
            return [x1]

        elementFinepIndexMap = util.extractElementFine(NWorldCoarse,
                                                       NCoarseElement,
                                                       0*NCoarseElement,
                                                       extractElements=False)
        elementFinetIndexMap = util.extractElementFine(NWorldCoarse,
                                                       NCoarseElement,
                                                       0*NCoarseElement,
                                                       extractElements=True)
        # Let lambdas = functions
        lambdasList = [f[elementFinepIndexMap] for f in computeFunctions()]

        ## Case
        # aPatch = 1
        # Let corrector Q = functions
        # Expect: muTPrime for first element T is 0, the others 1
        correctorsList = computeFunctions()
        aPatch = np.ones(world.NpFine)
        csi = lod.computeCoarseQuantities(patch, lambdasList, correctorsList, aPatch)

        self.assertAlmostEqual(np.sum(csi.muTPrime), 6*5*4-1)
        self.assertAlmostEqual(csi.muTPrime[0], 0)
        
        ## Case
        # aPatch = 1
        # Let corrector Q = 2*functions
        # Expect: muTPrime is 1 for first element and 4 for all others
        correctorsList = [2*f for f in computeFunctions()]
        aPatch = np.ones(world.NpFine)
        csi = lod.computeCoarseQuantities(patch, lambdasList, correctorsList, aPatch)

        self.assertAlmostEqual(np.sum(csi.muTPrime), 4*(6*5*4-1)+1)
        self.assertAlmostEqual(csi.muTPrime[0], 1)
        
        
    def test_computeSingleT(self):
        NWorldCoarse = np.array([4, 5, 6])
        NCoarseElement = np.array([5, 2, 3])
        world = World(NWorldCoarse, NCoarseElement)
        d = np.size(NWorldCoarse)
        
        k = 1
        iElementWorldCoarse = np.array([2, 1, 2])
        TInd = util.convertpCoordIndexToLinearIndex(NWorldCoarse, iElementWorldCoarse)
        patch = Patch(world, k, TInd)

        IPatch = interp.L2ProjectionPatchMatrix(patch)
        
        NtPatch = patch.NtFine

        aPatch = np.ones(NtPatch)
        basisCorrectorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)

        correctorSum = reduce(np.add, basisCorrectorsList)
        self.assertTrue(np.allclose(correctorSum, 0))

        csi = lod.computeBasisCoarseQuantities(patch, basisCorrectorsList, aPatch)
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
        k = np.max(NWorldCoarse)
        IWorld = interp.nodalPatchMatrix(Patch(world, k, 0))
        aWorld = np.exp(np.random.rand(world.NtFine))

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
        NCoarseElement = np.array([10, 10])
        world = World(NPatchCoarse, NCoarseElement)
        patch = Patch(world, 4, 0)
            
        NPatchFine = NPatchCoarse*NCoarseElement
        NpFine = np.prod(NPatchFine + 1)
        
        APatchFull = fem.assemblePatchMatrix(NPatchCoarse*NCoarseElement, world.ALocFine)
        bPatchFullList = [np.ones(NpFine)]

        fixed = util.boundarypIndexMap(NPatchFine)
        
        for IPatch in [interp.L2ProjectionPatchMatrix(patch),
                       interp.nodalPatchMatrix(patch)]:

            schurComplementSolver = lod.SchurComplementSolver()
            schurComplementSolution = lod.ritzProjectionToFinePatch(patch,
                                                                    APatchFull, bPatchFullList,
                                                                    IPatch,
                                                                    schurComplementSolver)[0]
            self.assertTrue(np.isclose(np.max(np.abs(schurComplementSolution[fixed])), 0))

class errorIndicators_TestCase(unittest.TestCase):
    def test_computeErrorIndicatorFine_zeroT(self):
        ## Setup
        # 2D, variables x0 and x1
        NCoarse = np.array([4, 3])
        NCoarseElement = np.array([2, 3])
        world = World(NCoarse, NCoarseElement)
        patch = Patch(world, 4, 0)
        NFine = NCoarse*NCoarseElement

        # Let functions = [x1, 2*x2]
        def computeFunctions():
            pc = util.pCoordinates(NFine)
            x1 = pc[:,0]
            x2 = pc[:,1]
            return [x1, 2*x2]

        # Mock corrector Q = functions
        correctorsList = computeFunctions()

        elementFinepIndexMap = util.extractElementFine(NCoarse,
                                                       NCoarseElement,
                                                       0*NCoarseElement,
                                                       extractElements=False)
        elementFinetIndexMap = util.extractElementFine(NCoarse,
                                                       NCoarseElement,
                                                       0*NCoarseElement,
                                                       extractElements=True)

        # Let lambdas = functions too
        lambdasList = [f[elementFinepIndexMap] for f in computeFunctions()]
        
        ## Case
        # AOld = ANew = scalar 1
        # Expect: Error indicator should be zero

        aOld = np.ones(world.NtFine, dtype=np.float64)
        aNew = aOld
        
        self.assertEqual(lod.computeErrorIndicatorFine(patch, lambdasList, correctorsList, aOld, aNew), 0)
       
        ## Case
        # AOld = scalar 1
        # ANew = scalar 10
        # Expect: Error indicator is sqrt of integral over 11 elements with value (10-1)**2/10**2

        aOld = np.ones(world.NtFine, dtype=np.float64)
        aNew = 10*aOld

        self.assertAlmostEqual(lod.computeErrorIndicatorFine(patch, lambdasList, correctorsList, aOld, aNew),
                               np.sqrt(11*(10-1)**2/10**2))
       
        ## Case
        # AOld = scalar 1
        # ANew = scalar 10 except in T where ANew = 1000
        # Expect: Error indicator is like in previous case, but /10

        aOld = np.ones(world.NtFine, dtype=np.float64)
        aNew = 10*aOld
        aNew[elementFinetIndexMap] = 1000

        self.assertAlmostEqual(lod.computeErrorIndicatorFine(patch, lambdasList, correctorsList, aOld, aNew),
                               0.1*np.sqrt(11*(10-1)**2/10**2))

    def test_computeErrorIndicatorCoarseFromCoefficients(self):
        ## Setup
        # 2D, variables x0 and x1
        NCoarse = np.array([4, 3])
        NCoarseElement = np.array([2, 3])
        world = World(NCoarse, NCoarseElement)
        patch = Patch(world, 4, 0)
        NFine = NCoarse*NCoarseElement
        NtCoarse = world.NtCoarse
        
        # muTPrime = 1, ..., NtCoarse
        muTPrime = np.arange(NtCoarse) + 1

        ## Case
        # aOld = aNew = 1
        # Expect: 0 error indicator
        aOld = np.ones(world.NtFine, dtype=np.float64)
        aNew = aOld

        self.assertAlmostEqual(lod.computeErrorIndicatorCoarseFromCoefficients(patch, muTPrime, aOld, aNew), 0)

        #### Same test for Matrix valued ####
        Aeye = np.tile(np.eye(2), [np.prod(NFine), 1, 1])
        aNew = np.einsum('tji, t -> tji', Aeye, aNew)

        self.assertAlmostEqual(lod.computeErrorIndicatorCoarseFromCoefficients(patch, muTPrime, aOld, aNew), 0)

        ## Case
        # aOld = 1
        # aNew = 10
        # Expect: sqrt(1/10 * 1/10*(10-1)**2*1 * (NtCoarse)*(NtCoarse+1)/2)
        aOld = np.ones(world.NtFine, dtype=np.float64)
        aNew = 10*aOld

        self.assertAlmostEqual(lod.computeErrorIndicatorCoarseFromCoefficients(patch, muTPrime, aOld, aNew),
                               np.sqrt(1/10 * 1/10*(10-1)**2*1 * (NtCoarse)*(NtCoarse+1)/2))

        #### Same test for Matrix valued ####
        aNew = np.einsum('tji, t -> tji', Aeye, aNew)
        aOld = np.einsum('tji, t-> tji', Aeye, aOld)

        self.assertAlmostEqual(lod.computeErrorIndicatorCoarseFromCoefficients(patch, muTPrime, aOld, aNew),
                               np.sqrt(1/10 * 1/10*(10-1)**2*1 * (NtCoarse)*(NtCoarse+1)/2))
        
        ## Case
        # aOld = 1
        # aNew = 1 except in TInd=2 (where muTPrime == 3), where it is 10
        # Expect: sqrt(1 * 1/10*(10-1)**2*1 * 3)
        aOld = np.ones(world.NtFine, dtype=np.float64)
        aNew = np.ones(world.NtFine, dtype=np.float64)
        
        elementFinetIndexMap = util.extractElementFine(NCoarse,
                                                       NCoarseElement,
                                                       np.array([2, 0]),
                                                       extractElements=True)
        aNew[elementFinetIndexMap] = 10

        self.assertAlmostEqual(lod.computeErrorIndicatorCoarseFromCoefficients(patch, muTPrime, aOld, aNew),
                               np.sqrt(1 * 1/10*(10-1)**2*1 * 3))

        #### Same test for Matrix valued ####
        aOld = np.einsum('tji, t-> tji', Aeye, aOld)

        self.assertAlmostEqual(lod.computeErrorIndicatorCoarseFromCoefficients(patch, muTPrime, aOld, aNew),
                               np.sqrt(1 * 1/10*(10-1)**2*1 * 3))

if __name__ == '__main__':
    unittest.main()
