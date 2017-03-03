import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
from itertools import count
import os

from pyevtk.hl import imageToVTK 

from gridlod import pg, interp, coef, util, fem, world, linalg, femsolver, transport
from gridlod.world import World

def saveCube(name, data, shape):
    uCube = np.reshape(data, shape[::-1])
    uCube = np.ascontiguousarray(np.transpose(uCube, axes=[2, 1, 0]))
    imageToVTK(name, pointData = {"u" : uCube} )


class PetrovGalerkinLOD_TestCase(unittest.TestCase):
    def test_alive(self):
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

        K = pglod.assembleMsStiffnessMatrix()
        self.assertTrue(np.all(K.shape == NpCoarse))

        basisCorrectors = pglod.assembleBasisCorrectors()
        self.assertTrue(np.all(basisCorrectors.shape == (NpFine, NpCoarse)))

    def test_stiffessMatrix(self):
        # Compare stiffness matrix from PG object with the one
        # computed from correctors and fine stiffness matrix
        NWorldFine = np.array([10, 10])
        NpFine = np.prod(NWorldFine+1)
        NtFine = np.prod(NWorldFine)
        NWorldCoarse = np.array([2, 2])
        NCoarseElement = NWorldFine/NWorldCoarse
        NtCoarse = np.prod(NWorldCoarse)
        NpCoarse = np.prod(NWorldCoarse+1)
        
        world = World(NWorldCoarse, NCoarseElement)

        np.random.seed(0)

        aBase = np.random.rand(NtFine)
        aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)
        
        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement) 
        IWorld = IPatchGenerator(0*NWorldCoarse, NWorldCoarse)

        k = 2
        printLevel = 0
        pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 0, printLevel)

        pglod.updateCorrectors(aCoef, clearFineQuantities=False)

        KmsFull = pglod.assembleMsStiffnessMatrix()
        KFull = pglod.assembleStiffnessMatrix()

        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        basisCorrectors = pglod.assembleBasisCorrectors()

        self.assertTrue(np.isclose(np.linalg.norm(IWorld*basisCorrectors.todense()), 0))
        
        modifiedBasis = basis - basisCorrectors
        AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aBase)
        
        KmsRef = np.dot(basis.T, AFine*modifiedBasis)
        KRef = np.dot(basis.T, AFine*basis)
 
        self.assertTrue(np.isclose(np.linalg.norm(KFull.todense()-KRef.todense()), 0))
        self.assertTrue(np.isclose(np.linalg.norm(KmsFull.todense()-KmsRef.todense()), 0))
        
    def test_1d(self):
        # Example from Peterseim, Variational Multiscale Stabilization and the Exponential Decay of correctors, p. 2
        # Two modifications: A with minus and u(here) = 1/4*u(paper).
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
            boundaryConditions = np.array([[0, 0]])
            world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
            
            xpCoarse = util.pCoordinates(NWorldCoarse).flatten()
            
            NpCoarse = np.prod(NWorldCoarse+1)

            IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, boundaryConditions)
            aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aFine)

            pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 0)
            pglod.updateCorrectors(aCoef, clearFineQuantities=False)

            KFull = pglod.assembleMsStiffnessMatrix()
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

            AFine = fem.assemblePatchMatrix(NFine, world.ALocFine, aFine)
            MFine = fem.assemblePatchMatrix(NFine, world.MLocFine)

            newErrorCoarse = np.sqrt(np.dot(uSol - uLodCoarse, MFine*(uSol - uLodCoarse)))
            newErrorFine = np.sqrt(np.dot(uSol - uLodFine, AFine*(uSol - uLodFine)))

            self.assertTrue(newErrorCoarse < previousErrorCoarse)
            self.assertTrue(newErrorFine < previousErrorFine)

    def test_1d_toReference(self):
        NWorldFine = np.array([200])
        NWorldCoarse = np.array([10])
        NCoarseElement = NWorldFine/NWorldCoarse
        boundaryConditions = np.array([[0, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        NpFine = np.prod(NWorldFine+1)
        NtFine = np.prod(NWorldFine)
        NpCoarse = np.prod(NWorldCoarse+1)
        
        aBase = np.zeros(NtFine)
        aBase[:90] = 1
        aBase[90:] = 2
        
        k = 10
            
        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, boundaryConditions)
        aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)

        pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 0)
        pglod.updateCorrectors(aCoef, clearFineQuantities=False)

        KmsFull = pglod.assembleMsStiffnessMatrix()
        KFull = pglod.assembleStiffnessMatrix()
        MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)

        free  = util.interiorpIndexMap(NWorldCoarse)

        coords = util.pCoordinates(NWorldCoarse)
        g = 1-coords[:,0]
        bFull = -KmsFull*g

        KmsFree = KmsFull[free][:,free]
        bFree = bFull[free]

        xFree = sparse.linalg.spsolve(KmsFree, bFree)

        basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        basisCorrectors = pglod.assembleBasisCorrectors()
        modifiedBasis = basis - basisCorrectors
        xFull = np.zeros(NpCoarse)
        xFull[free] = xFree
        uLodCoarse = basis*(xFull + g)
        uLodFine = modifiedBasis*(xFull + g)

        coordsFine = util.pCoordinates(NWorldFine)
        gFine = 1-coordsFine[:,0]
        uFineFull, AFine, _ = femsolver.solveFine(world, aBase, None, -gFine, boundaryConditions)
        uFineFull += gFine

        errorFine = np.sqrt(np.dot(uFineFull - uLodFine, AFine*(uFineFull - uLodFine)))
        self.assertTrue(np.isclose(errorFine, 0))

        # Also compute upscaled solution and compare with uFineFull
        uLodFineUpscaled = basis*(xFull + g) - pglod.computeCorrection(ARhsFull=basis*(xFull + g))
        self.assertTrue(np.allclose(uLodFineUpscaled, uLodFine))
        
    def test_2d_exactSolution(self):
        NWorldFine = np.array([30, 40])
        NpFine = np.prod(NWorldFine+1)
        NtFine = np.prod(NWorldFine)
        NWorldCoarse = np.array([3, 4])
        NCoarseElement = NWorldFine/NWorldCoarse
        NtCoarse = np.prod(NWorldCoarse)
        NpCoarse = np.prod(NWorldCoarse+1)
        
        boundaryConditions = np.array([[0, 0],
                                       [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        np.random.seed(0)
        
        aBaseSquare = np.exp(5*np.random.random_sample(NWorldFine[0]))
        aBaseCube = np.tile(aBaseSquare, [NWorldFine[1],1])
        aBaseCube = aBaseCube[...,np.newaxis]

        aBase = aBaseCube.flatten()

        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, boundaryConditions)

        rCoarse = np.ones(NtCoarse)
        aCoef = coef.coefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarse)
        
        k = 4
        printLevel = 0
        epsilonTol = 0.05
        pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, epsilonTol, printLevel)
        pglod.updateCorrectors(aCoef, clearFineQuantities=False)

        boundaryMap = boundaryConditions==0
        fixed = util.boundarypIndexMap(NWorldCoarse, boundaryMap)
        free = np.setdiff1d(np.arange(0,NpCoarse), fixed)
        
        
        coords = util.pCoordinates(NWorldCoarse)
        xC = coords[:,0]
        yC = coords[:,1]
        g = 1-xC

        firstIteration = True

        # First case is to not modify. Error should be 0
        # The other cases modify one, a few or half of the coarse elements to different degrees.
        rCoarseModPairs = [([], []), ([0], [2.]), ([10], [3.]), ([4, 3, 2], [1.3, 1.5, 1.8])]
        rCoarseModPairs.append((range(NtCoarse/2), [2]*NtCoarse))
        rCoarseModPairs.append((range(NtCoarse/2), [0.9]*NtCoarse))
        rCoarseModPairs.append((range(NtCoarse/2), [0.95]*NtCoarse))
        
        for i, rCoarseModPair in zip(count(), rCoarseModPairs):
            for ind, val in zip(rCoarseModPair[0], rCoarseModPair[1]):
                rCoarse[ind] *= val
                
            aCoef = coef.coefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarse)
            pglod.updateCorrectors(aCoef, clearFineQuantities=False)

            KmsFull = pglod.assembleMsStiffnessMatrix()
            bFull = -KmsFull*g

            KmsFree = KmsFull[free][:,free]

            bFree = bFull[free]
            xFree = sparse.linalg.spsolve(KmsFree, bFree)

            basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
            basisCorrectors = pglod.assembleBasisCorrectors()
            modifiedBasis = basis - basisCorrectors

            xFull = np.zeros(NpCoarse)
            xFull[free] = xFree
            uLodFine = modifiedBasis*(xFull + g)

            gFine = basis*g
            uFineFull, AFine, MFine = femsolver.solveFine(world, aCoef.aFine, None, -gFine, boundaryConditions)
            uFineFull += gFine

            errorFineA = np.sqrt(np.dot(uFineFull - uLodFine, AFine*(uFineFull - uLodFine)))
            errorFineM = np.sqrt(np.dot(uFineFull - uLodFine, MFine*(uFineFull - uLodFine)))
            if firstIteration:
                self.assertTrue(np.isclose(errorFineA, 0))
                self.assertTrue(np.isclose(errorFineM, 0))

                # Also compute upscaled solution and compare with uFineFull
                uLodFineUpscaled = basis*(xFull + g) - pglod.computeCorrection(ARhsFull=basis*(xFull + g))
                self.assertTrue(np.allclose(uLodFineUpscaled, uLodFine))
                
                firstIteration = False
            else:
                # For this problem, it seems that
                # error < 1.1*errorTol
                self.assertTrue(errorFineA <= 1.1*epsilonTol)


                
    def test_2d_flux(self):
        # Stripes perpendicular to flow direction gives effective
        # permeability as the harmonic mean of the permeabilities of
        # the stripes.
        NWorldFine = np.array([20, 20])
        NpFine = np.prod(NWorldFine+1)
        NtFine = np.prod(NWorldFine)
        NWorldCoarse = np.array([2, 2])
        NCoarseElement = NWorldFine/NWorldCoarse
        NtCoarse = np.prod(NWorldCoarse)
        NpCoarse = np.prod(NWorldCoarse+1)
        
        boundaryConditions = np.array([[0, 0],
                                       [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        np.random.seed(0)
        
        aBaseSquare = np.exp(5*np.random.random_sample(NWorldFine[1]))
        aBaseCube = np.tile(aBaseSquare[...,np.newaxis], [NWorldFine[0],1])
        aBaseCube = aBaseCube[...,np.newaxis]

        aBase = aBaseCube.flatten()

        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, boundaryConditions)
        
        aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)
        
        k = 2
        printLevel = 0
        pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 0, printLevel)
        pglod.updateCorrectors(aCoef)

        KmsFull = pglod.assembleMsStiffnessMatrix()

        coords = util.pCoordinates(NWorldCoarse)
        xC = coords[:,0]
        g = 1-xC
        bFull = -KmsFull*g

        boundaryMap = boundaryConditions==0
        fixed = util.boundarypIndexMap(NWorldCoarse, boundaryMap)
        free = np.setdiff1d(np.arange(0,NpCoarse), fixed)
        KmsFree = KmsFull[free][:,free]
        
        bFree = bFull[free]
        xFree = sparse.linalg.spsolve(KmsFree, bFree)
        xFull = np.zeros(NpCoarse)
        xFull[free] = xFree

        MGammaLocGetter = fem.localBoundaryMassMatrixGetter(NWorldCoarse)
        MGammaFull = fem.assemblePatchBoundaryMatrix(NWorldCoarse, MGammaLocGetter, boundaryMap=boundaryMap)
        
        # Solve (F, w) = a(u0, w) + a(g, w) in space of fixed DoFs only
        KmsFixedFull = KmsFull[fixed]
        cFixed = KmsFixedFull*(xFull + g)
        MGammaFixed = MGammaFull[fixed][:,fixed]
        FFixed = sparse.linalg.spsolve(MGammaFixed, cFixed)
        FFull = np.zeros(NpCoarse)
        FFull[fixed] = FFixed

        self.assertTrue(np.isclose(np.mean(FFixed[FFixed>0]), stats.hmean(aBaseSquare)))
        
    def test_3d(self):
        return
        NWorldFine = np.array([60, 220, 50])
        NpFine = np.prod(NWorldFine+1)
        NtFine = np.prod(NWorldFine)
        NWorldCoarse = np.array([6, 22, 5])
        NCoarseElement = NWorldFine/NWorldCoarse
        NtCoarse = np.prod(NWorldCoarse)
        NpCoarse = np.prod(NWorldCoarse+1)
        
        boundaryConditions = np.array([[1, 1],
                                       [0, 0],
                                       [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        
        aBase = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/data/upperness_x.txt')
        #aBase = aBase[::8]

        
        print 'a'
        coords = util.pCoordinates(NWorldFine)
        gFine = 1-coords[:,1]
        uFineFull, AFine, _ = femsolver.solveFine(world, aBase, None, -gFine, boundaryConditions)
        print 'b'
        
        rCoarse = np.ones(NtCoarse)
        
        self.assertTrue(np.size(aBase) == NtFine)

        if True:
            IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, boundaryConditions)
            aCoef = coef.coefficientCoarseFactor(NWorldCoarse, NCoarseElement, aBase, rCoarse)
        
            k = 2
            printLevel = 1
            pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 1e-1, printLevel)
            pglod.updateCorrectors(aCoef, clearFineQuantities=True)

            KmsFull = pglod.assembleMsStiffnessMatrix()

            coords = util.pCoordinates(NWorldCoarse)
            g = 1-coords[:,1]
            bFull = -KmsFull*g
            
            boundaryMap = boundaryConditions==0
            fixed = util.boundarypIndexMap(NWorldCoarse, boundaryMap)
            free = np.setdiff1d(np.arange(0,NpCoarse), fixed)

            KmsFree = KmsFull[free][:,free]
            bFree = bFull[free]

            xFree = sparse.linalg.spsolve(KmsFree, bFree)

            uCoarse = np.zeros(NpCoarse)
            uCoarse[free] = xFree
            uCoarse += g
            uCube = np.reshape(uCoarse, (NWorldCoarse+1)[::-1])
            uCube = np.ascontiguousarray(np.transpose(uCube, axes=[2, 1, 0]))
            
            imageToVTK("./image", pointData = {"u" : uCube} )

        if False:
            coord = util.pCoordinates(NWorldCoarse)
            uCoarse = coord[:,1].flatten()
            uCube = np.reshape(uCoarse, (NWorldCoarse+1)[::-1])
            uCube = np.ascontiguousarray(np.transpose(uCube, axes=[2, 1, 0]))
            imageToVTK("./image", pointData = {"u" : uCube} )
            
        # basis = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        # basisCorrectors = pglod.assembleBasisCorrectors()
        # modifiedBasis = basis - basisCorrectors
        # xFull = np.zeros(NpCoarse)
        # xFull[free] = xFree
        # uLodCoarse = basis*xFull
        # uLodFine = modifiedBasis*xFull

        #freeFine  = util.interiorpIndexMap(NWorldFine)
        #AFine = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aBase)
        #MFine = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)
        #bFine = MFine*np.ones(NpFine)
        #AFineFree = AFine[freeFine][:,freeFine]
        #bFineFree = bFine[freeFine]
        #uFineFree = linalg.linSolve(AFine, bFine)
        #uFineFull = np.zeros(NpFine)
        #uFineFull[freeFine] = uFineFree

        #np.savetxt('uFineFull', uFineFull)
        
    def test_00coarseFlux(self):
        NWorldFine = np.array([20, 20])
        NpFine = np.prod(NWorldFine+1)
        NtFine = np.prod(NWorldFine)
        NWorldCoarse = np.array([4, 4])
        NCoarseElement = NWorldFine/NWorldCoarse
        NtCoarse = np.prod(NWorldCoarse)
        NpCoarse = np.prod(NWorldCoarse+1)
        
        boundaryConditions = np.array([[0, 0],
                                       [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        np.random.seed(0)
        
        aBaseSquare = np.exp(5*np.random.random_sample(NWorldFine[1]))
        aBaseCube = np.tile(aBaseSquare[...,np.newaxis], [NWorldFine[0],1])
        aBaseCube = aBaseCube[...,np.newaxis]

        aBase = aBaseCube.flatten()

        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, NWorldCoarse, NCoarseElement, boundaryConditions)
        
        aCoef = coef.coefficientFine(NWorldCoarse, NCoarseElement, aBase)
        
        k = 4
        printLevel = 0
        pglod = pg.PetrovGalerkinLOD(world, k, IPatchGenerator, 0, printLevel)
        pglod.updateCorrectors(aCoef)

        KmsFull = pglod.assembleMsStiffnessMatrix()

        coords = util.pCoordinates(NWorldCoarse)
        xC = coords[:,0]
        g = 1-xC
        bFull = -KmsFull*g

        boundaryMap = boundaryConditions==0
        fixed = util.boundarypIndexMap(NWorldCoarse, boundaryMap)
        free = np.setdiff1d(np.arange(0,NpCoarse), fixed)
        KmsFree = KmsFull[free][:,free]
        
        bFree = bFull[free]
        xFree = sparse.linalg.spsolve(KmsFree, bFree)
        xFull = np.zeros(NpCoarse)
        xFull[free] = xFree
        uFull = xFull+g

        lodFluxTF = pglod.computeFaceFluxTF(uFull)

        ## Compute fine solution
        coords = util.pCoordinates(NWorldFine)
        xF = coords[:,0]
        gFine = 1-xF
        uFineFull, AFine, _ = femsolver.solveFine(world, aBase, None, -gFine, boundaryConditions)
        uFineFull = uFineFull+gFine
        fineFluxTF = transport.computeHarmonicMeanFaceFlux(NWorldCoarse, NWorldCoarse, NCoarseElement, aBase, uFineFull)

        self.assertTrue(np.allclose(lodFluxTF, fineFluxTF, rtol=1e-7))
        
if __name__ == '__main__':
    #import cProfile
    #command = """unittest.main()"""
    #cProfile.runctx( command, globals(), locals(), filename="test_pg.profile" )
    unittest.main()

