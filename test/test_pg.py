import unittest
import numpy as np
import scipy.sparse as sparse
import os

import pg
from world import World
import interp
import coef
import util
import fem
import linalg

class PetrovGalerkinLOD_TestCase(unittest.TestCase):
    def test_alive(self):
        return
        NWorldCoarse = np.array([5,5])
        NCoarseElement = np.array([20,20])
        NFine = NWorldCoarse*NCoarseElement
        NtFine = np.prod(NFine)
        NpCoarse = np.prod(NWorldCoarse+1)
        NpFine = np.prod(NWorldCoarse*NCoarseElement+1)
        
        world = World(NWorldCoarse, NCoarseElement)

        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement)
        
        k = 5
        
        pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 0)

        aBase = np.ones(NtFine)
        pglod.updateCorrectors(coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase), clearFineQuantities=False)

        K = pglod.assembleStiffnessMatrix()
        self.assertTrue(np.all(K.shape == NpCoarse))

        basisCorrectors = pglod.assembleBasisCorrectors()
        self.assertTrue(np.all(basisCorrectors.shape == (NpFine, NpCoarse)))

    def test_1d(self):
        # Example from Peterseim, Variational Multiscale Stabilization and the Exponential Decay of correctors, p. 2
        # Two modifications: A with minus and u(here) = 1/4*u(paper).
        return
    
        NFine = np.array([3200])
        NpFine = np.prod(NFine+1)
        NList = [10, 20, 40, 80, 160]
        epsilon = 1024./NFine
        epsilon = 1./320
        k = 2
        
        pi = np.pi
        
        xt = util.tCoordinates(NFine).flatten()
        xp = util.pCoordinates(NFine).flatten()
        #aFine = (2 + np.cos(2*pi*xt/epsilon))**(-1)
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
            NCoarseElement = NFine/NWorldCoarse
            world = World(NWorldCoarse, NCoarseElement)
            
            xpCoarse = util.pCoordinates(NWorldCoarse).flatten()
            
            NpCoarse = np.prod(NWorldCoarse+1)

            IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement)
            aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine)

            pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 0)
            pglod.updateCorrectors(aCoef, clearFineQuantities=False)

            KFull = pglod.assembleStiffnessMatrix()
            MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)

            free  = util.interiorpIndexMap(NWorldCoarse)

            f = np.ones(NpCoarse)
            bFull = MFull*f

            KFree = KFull[free][:,free]
            bFree = bFull[free]

            xFree = sparse.linalg.spsolve(KFree, bFree)

            basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
            basisCorrectors = pglod.assembleBasisCorrectors()
            modifiedBasis = basis - basisCorrectors
            xFull = np.zeros(NpCoarse)
            xFull[free] = xFree
            uLodCoarse = basis*xFull
            uLodFine = modifiedBasis*xFull

            # freeFine  = util.interiorpIndexMap(NFine)
            AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aFine)
            MFine = fem.assemblePatchMatrix(NFine, world.MLocFine)
            # bFine = MFine*np.ones(NpFine)
            # AFineFree = AFine[freeFine][:,freeFine]
            # bFineFree = bFine[freeFine]
            # uFineFree = sparse.linalg.spsolve(AFineFree, bFineFree)
            # uFineFull = np.zeros(NpFine)
            # uFineFull[freeFine] = uFineFree

            #import matplotlib.pyplot as plt
            #plt.plot(xpCoarse, xFull, 'r')
            #plt.plot(xp, uSol-uLodFine)
            #plt.plot(xp, np.log10(np.abs(uFineFull-uSol)), 'k')
            #plt.plot(xp, np.log10(np.abs(uFineFull-uLodFine)), 'r')
            #plt.plot(xp, np.log10(np.abs(uSol-uLodFine)), 'b')
            #plt.plot(xp, uFineFull-uLodFine)
            #plt.plot(xp, uLodFine, 'r')
            #plt.plot(xp, uFineFull, 'b')
            #plt.plot(xp, uSol, 'k')
            #plt.plot(xt, aFine, 'k')
            #plt.show()
            
            #print np.sqrt(np.dot(uSol - uFineFull, AFine*(uSol - uFineFull)))
            #print np.sqrt(np.dot(uFineFull - uLodFine, AFine*(uFineFull - uLodFine)))
            newErrorCoarse = np.sqrt(np.dot(uSol - uLodCoarse, MFine*(uSol - uLodCoarse)))
            newErrorFine = np.sqrt(np.dot(uSol - uLodFine, AFine*(uSol - uLodFine)))

            self.assertTrue(newErrorCoarse < previousErrorCoarse)
            self.assertTrue(newErrorFine < previousErrorFine)

    def test_3d(self):
        NWorldFine = np.array([60, 220, 50])
        NpFine = np.prod(NWorldFine+1)
        NtFine = np.prod(NWorldFine)
        NWorldCoarse = np.array([6, 22, 5])
        NCoarseElement = NWorldFine/NWorldCoarse
        NtCoarse = np.prod(NWorldCoarse)
        
        world = World(NWorldCoarse, NCoarseElement)

        aBase = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/data/upperness_x.txt')
        #aBase = aBase[::8]
        rCoarse = np.ones(NtCoarse)
        
        self.assertTrue(np.size(aBase) == NtFine)

        if False:
            IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement)
            aCoef = coef.coefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarse)
        
            k = 1
            printLevel = 1
            pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 1e-1, printLevel)
            pglod.updateCorrectors(aCoef, clearFineQuantities=True)

            rCoarse2 = np.array(rCoarse)
            rCoarse2[10] = 1.2
            aCoef2 = coef.coefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarse2)
            pglod.updateCorrectors(aCoef2, clearFineQuantities=True)

        
        # KFull = pglod.assembleStiffnessMatrix()
        # MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)

        # free  = util.interiorpIndexMap(NWorldCoarse)

        # f = np.ones(NpCoarse)
        # bFull = MFull*f

        # KFree = KFull[free][:,free]
        # bFree = bFull[free]

        
        # xFree = sparse.linalg.spsolve(KFree, bFree)

        # basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        # basisCorrectors = pglod.assembleBasisCorrectors()
        # modifiedBasis = basis - basisCorrectors
        # xFull = np.zeros(NpCoarse)
        # xFull[free] = xFree
        # uLodCoarse = basis*xFull
        # uLodFine = modifiedBasis*xFull

        freeFine  = util.interiorpIndexMap(NWorldFine)
        AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aBase)
        MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)
        bFine = MFine*np.ones(NpFine)
        AFineFree = AFine[freeFine][:,freeFine]
        bFineFree = bFine[freeFine]
        uFineFree = linalg.linSolve(AFine, bFine)
        uFineFull = np.zeros(NpFine)
        uFineFull[freeFine] = uFineFree

        np.savetxt('uFineFull', uFineFull)
        
if __name__ == '__main__':
    import cProfile
    command = """unittest.main()"""
    cProfile.runctx( command, globals(), locals(), filename="test_pg.profile" )
    #unittest.main()

