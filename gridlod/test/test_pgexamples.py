import unittest
import numpy as np
import scipy.sparse as sparse

from gridlod import pglod, util, lod, interp, coef, fem
from gridlod.world import World, Patch

class exampleProblem_TestCase(unittest.TestCase):
    def test_1d(self):
        def test_1d_core(mapper):
            # Example from Peterseim, Variational Multiscale Stabilization and the Exponential Decay of correctors, p. 2
            # Two modifications: A with minus and u(here) = 1/4*u(paper).
            NFine = np.array([3200])
            NpFine = np.prod(NFine+1)
            NList = [10, 20, 40, 80, 160]
            epsilon = 1./320
            k = 2

            pi = np.pi

            xt = util.tCoordinates(NFine).flatten()
            xp = util.pCoordinates(NFine).flatten()
            aFine = (2 - np.cos(2*pi*xt/epsilon))**(-1)

            uSol  = 4*(xp - xp**2) - 4*epsilon*(1/(4*pi)*np.sin(2*pi*xp/epsilon) -
                                                1/(2*pi)*xp*np.sin(2*pi*xp/epsilon) -
                                                epsilon/(4*pi**2)*np.cos(2*pi*xp/epsilon) +
                                                epsilon/(4*pi**2))

            uSol = uSol/4

            previousErrorCoarse = np.inf
            previousErrorFine = np.inf

            for N in NList:
                NWorldCoarse = np.array([N])
                NCoarseElement = NFine//NWorldCoarse
                boundaryConditions = np.array([[0, 0]])
                world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

                xpCoarse = util.pCoordinates(NWorldCoarse).flatten()

                def computeKmsij(TInd):
                    patch = Patch(world, k, TInd)
                    IPatch = lambda: interp.L2ProjectionPatchMatrix(patch.iPatchWorldCoarse, patch.NPatchCoarse, NWorldCoarse, NCoarseElement, boundaryConditions)
                    aPatch = lambda: coef.localizeCoefficient(patch, aFine)

                    correctorsList = lod.computeBasisCorrectors(patch, IPatch, aPatch)
                    csi = lod.computeCoarseQuantities(patch, correctorsList, aPatch)
                    return patch, correctorsList, csi.Kmsij

                # Use mapper to distribute computations (mapper could be the 'map' built-in or e.g. an ipyparallel map)
                patchT, correctorsListT, KmsijT = zip(*mapper(computeKmsij, range(world.NtCoarse)))

                KFull = pglod.assembleMsStiffnessMatrix(world, patchT, KmsijT)
                MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)

                free  = util.interiorpIndexMap(NWorldCoarse)

                f = np.ones(world.NpCoarse)
                bFull = MFull*f

                KFree = KFull[free][:,free]
                bFree = bFull[free]

                xFree = sparse.linalg.spsolve(KFree, bFree)

                basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
                basisCorrectors = pglod.assembleBasisCorrectors(world, patchT, correctorsListT)
                modifiedBasis = basis - basisCorrectors
                xFull = np.zeros(world.NpCoarse)
                xFull[free] = xFree
                uLodCoarse = basis*xFull
                uLodFine = modifiedBasis*xFull

                AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aFine)
                MFine = fem.assemblePatchMatrix(NFine, world.MLocFine)

                newErrorCoarse = np.sqrt(np.dot(uSol - uLodCoarse, MFine*(uSol - uLodCoarse)))
                newErrorFine = np.sqrt(np.dot(uSol - uLodFine, AFine*(uSol - uLodFine)))

                self.assertTrue(newErrorCoarse < previousErrorCoarse)
                self.assertTrue(newErrorFine < previousErrorFine)
                
        test_1d_core(map)

        # Although disabled in the unit test, the code below shows an example
        # of how to use an ipyparallel cluster to compute the correctors.
        if False:
            import ipyparallel as ipp
            client = ipp.Client()
            client[:].use_cloudpickle()
            view = client.load_balanced_view()
            test_1d_core(view.map_sync)
            

if __name__ == '__main__':
    unittest.main()
