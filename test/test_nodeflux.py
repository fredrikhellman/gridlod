import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from gridlod import nodeflux, fem, util, world
from gridlod.world import World

class computeBoundaryFlux_TestCase(unittest.TestCase):

    def test_computeBoundaryFlux_1d(self):
        NWorldCoarse = np.array([10])
        NCoarseElement = np.array([1])
        boundaryConditions = np.array([[1, 0]])
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NWorldFine = NWorldCoarse*NCoarseElement
        
        coords = util.pCoordinates(NWorldFine)

        Np = np.prod(NWorldCoarse+1)
        
        fFull = -coords[:,0]
        AFull = fem.assemblePatchMatrix(NWorldCoarse, world.ALocCoarse)
        A = AFull[:-1][:,:-1]
        MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
        bFull = MFull*fFull
        b = bFull[:-1]
        
        u = sparse.linalg.spsolve(A, b)
        uFull = np.zeros(Np)
        uFull[:-1] = u

        ROmega = AFull*uFull - bFull
    
        sigma = nodeflux.computeBoundaryFlux(world, ROmega)
        self.assertTrue(np.size(sigma) == Np)
        self.assertTrue(np.isclose(sigma[-1], 0.5))
        self.assertTrue(np.isclose(sigma[-1], NWorldCoarse[0]*(uFull[-1]-uFull[-2]), atol=1./NWorldCoarse[0]))
        
    def test_computeBoundaryFlux_2d(self):
        NWorldCoarse = np.array([10, 10])
        NCoarseElement = np.array([1, 1])
        boundaryConditions = np.array([[0, 0], [1, 1]])
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NWorldFine = NWorldCoarse*NCoarseElement
        
        coords = util.pCoordinates(NWorldFine)
        uCoarse = coords[:,0] + (coords[:,0]>0.5)*coords[:,0]

        Np = np.prod(NWorldCoarse+1)
                
        ACoarseFull = fem.assemblePatchMatrix(NWorldCoarse, world.ALocCoarse)

        ROmega = ACoarseFull*uCoarse
    
        sigma = nodeflux.computeBoundaryFlux(world, ROmega)

        boundaryMap = boundaryConditions==0
        dirichletNodes = util.boundarypIndexMap(NWorldCoarse, boundaryMap)
        otherNodes = np.setdiff1d(np.arange(Np), dirichletNodes)
        self.assertTrue(np.allclose(sigma[dirichletNodes][::2], -1.))
        self.assertTrue(np.allclose(sigma[dirichletNodes][1::2], 2.))
        self.assertTrue(np.allclose(sigma[otherNodes], 0.))
    
    def test_computeCoarseElementFlux_1d(self):
        NWorldCoarse = np.array([10])
        NCoarseElement = np.array([1])
        boundaryConditions = np.array([[1, 0]])
        d = 1
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        NWorldFine = NWorldCoarse*NCoarseElement
        
        coords = util.pCoordinates(NWorldFine)

        Np = np.prod(NWorldCoarse+1)
        Nt = np.prod(NWorldCoarse)
        
        fFull = -coords[:,0]
        AFull = fem.assemblePatchMatrix(NWorldCoarse, world.ALocCoarse)
        A = AFull[:-1][:,:-1]
        MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
        bFull = MFull*fFull
        b = bFull[:-1]
        
        u = sparse.linalg.spsolve(A, b)
        uFull = np.zeros(Np)
        uFull[:-1] = u

        RT = np.zeros([2**d, Nt])
        for T in np.arange(Nt):
            RT[:,T] = np.dot(world.ALocCoarse, np.array([uFull[T], uFull[T+1]])) - \
                      np.dot(world.MLocCoarse, np.array([fFull[T], fFull[T+1]]))
    
        sigmaFluxT, _ = nodeflux.computeCoarseElementFlux(world, RT)

        self.assertTrue(np.allclose(sigmaFluxT[1:,0], -sigmaFluxT[0:-1, 1]))
        self.assertTrue(np.allclose(sigmaFluxT[:,0]+sigmaFluxT[:,1], -0.5*(fFull[:-1]+fFull[1:])/Nt))
        
    def test_computeCoarseElementFlux_2d(self):
        NWorldCoarse = np.array([80, 80])
        NCoarseElement = np.array([1, 1])
        boundaryConditions = np.array([[0, 0],
                                       [0, 0]])
        d = 2
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        coords = util.pCoordinates(NWorldCoarse)
        x = coords[:,0]
        y = coords[:,1]

        aCoef = 1
        bCoef = 1
        
        uRefFull = np.sin(2*np.pi*aCoef*x)*np.sin(2*np.pi*bCoef*y)
        fFull = ((2*np.pi*aCoef)**2 + (2*np.pi*bCoef)**2)*uRefFull

        Np = np.prod(NWorldCoarse+1)
        Nt = np.prod(NWorldCoarse)

        free = util.interiorpIndexMap(NWorldCoarse)
        fixed = util.boundarypIndexMap(NWorldCoarse)
        
        AFull = fem.assemblePatchMatrix(NWorldCoarse, world.ALocCoarse)
        A = AFull[free][:,free]
        MFull = fem.assemblePatchMatrix(NWorldCoarse, world.MLocCoarse)
        bFull = MFull*fFull
        b = bFull[free]
        
        uFree = sparse.linalg.spsolve(A, b)
        uFull = np.zeros(Np)
        uFull[free] = uFree
    
        RT = np.zeros([2**d, Nt])
        TpIndex = util.elementpIndexMap(NWorldCoarse)
        TpStart = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)
        for T in np.arange(Nt):
            RT[:,T] = np.dot(world.ALocCoarse, uFull[TpStart[T] + TpIndex]) - \
                      np.dot(world.MLocCoarse, fFull[TpStart[T] + TpIndex])
    
        sigmaFluxT, nodeFluxT = nodeflux.computeCoarseElementFlux(world, RT)
        nodeFluxes = np.zeros(Np)
        for T in np.arange(Nt):
            nodeFluxes[TpStart[T] + TpIndex] += nodeFluxT[T,:]

        self.assertTrue(np.allclose(nodeFluxes[free], 0))
            
        ROmega = AFull*uFull - bFull
        boundaryFlux = nodeflux.computeBoundaryFlux(world, ROmega)

        boundaryFluxRefx0 = -(1.*(x < 0+1e-7))*(2*np.pi*aCoef*np.cos(2*np.pi*aCoef*x)*np.sin(2*np.pi*bCoef*y))
        boundaryFluxRefx1 =  (1.*(x > 1-1e-7))*(2*np.pi*aCoef*np.cos(2*np.pi*aCoef*x)*np.sin(2*np.pi*bCoef*y))
        boundaryFluxRefy0 = -(1.*(y < 0+1e-7))*(2*np.pi*bCoef*np.sin(2*np.pi*aCoef*x)*np.cos(2*np.pi*bCoef*y))
        boundaryFluxRefy1 =  (1.*(y > 1-1e-7))*(2*np.pi*bCoef*np.sin(2*np.pi*aCoef*x)*np.cos(2*np.pi*bCoef*y))
        boundaryFluxRef = boundaryFluxRefx0 + boundaryFluxRefx1 + boundaryFluxRefy0 + boundaryFluxRefy1
        boundaryFluxRef[free] = 0
        relativeDiscreteErrorNorm = np.linalg.norm((boundaryFlux - boundaryFluxRef)[fixed])/np.linalg.norm(boundaryFlux[fixed])
        self.assertTrue(np.isclose(relativeDiscreteErrorNorm, 0, atol=1e-2))

        MLocGetter = fem.localBoundaryMassMatrixGetter(NWorldCoarse)
        MBoundaryFull = fem.assemblePatchBoundaryMatrix(NWorldCoarse,
                                                        MLocGetter)
        self.assertTrue(np.isclose(np.max(np.abs(MBoundaryFull*boundaryFluxRef-nodeFluxes)), 0, atol=1e-2))

    def test_computeCoarseElementFlux_2d_twoScales(self):
        NWorldCoarse = np.array([20, 20])
        NCoarseElement = np.array([5, 5])
        NWorldFine = NWorldCoarse*NCoarseElement
        boundaryConditions = np.array([[0, 0],
                                       [0, 0]])
        d = 2
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        coordsFine = util.pCoordinates(NWorldFine)
        xFine = coordsFine[:,0]
        yFine = coordsFine[:,1]

        coordsCoarse = util.pCoordinates(NWorldCoarse)
        xCoarse = coordsCoarse[:,0]
        yCoarse = coordsCoarse[:,1]

        aCoef = 1
        bCoef = 1
        
        uRefFull = np.sin(2*np.pi*aCoef*xFine)*np.sin(2*np.pi*bCoef*yFine)
        fFull = ((2*np.pi*aCoef)**2 + (2*np.pi*bCoef)**2)*uRefFull

        NpFine = np.prod(NWorldFine+1)
        NpCoarse = np.prod(NWorldCoarse+1)
        NtCoarse = np.prod(NWorldCoarse)

        free = util.interiorpIndexMap(NWorldFine)
        
        AFull = fem.assemblePatchMatrix(NWorldFine, world.ALocFine)
        A = AFull[free][:,free]
        MFull = fem.assemblePatchMatrix(NWorldFine, world.MLocFine)
        bFull = MFull*fFull
        b = bFull[free]
        
        uFree = sparse.linalg.spsolve(A, b)
        uFull = np.zeros(NpFine)
        uFull[free] = uFree
    
        ATFull = fem.assemblePatchMatrix(NCoarseElement, world.ALocFine)
        MTFull = fem.assemblePatchMatrix(NCoarseElement, world.MLocFine)
        localBasis = world.localBasis

        ###
        RT = np.zeros([2**d, NtCoarse])
        TpIndex = util.lowerLeftpIndexMap(NCoarseElement, NWorldFine)
        TpStart = util.pIndexMap(NWorldCoarse-1, NWorldFine, NCoarseElement)
        for T in np.arange(NtCoarse):
            RT[:,T] = np.dot(localBasis.T,
                             ATFull*(uFull[TpStart[T] + TpIndex]) -
                             MTFull*(fFull[TpStart[T] + TpIndex]))
        ###
        
        sigmaFluxT, nodeFluxT = nodeflux.computeCoarseElementFlux(world, RT)

        ###
        nodeFluxes = np.zeros(NpCoarse)
        TpIndex = util.elementpIndexMap(NWorldCoarse)
        TpStart = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)
        for T in np.arange(NtCoarse):
            nodeFluxes[TpStart[T] + TpIndex] += nodeFluxT[T,:]
        ###
            
        freeCoarse = util.interiorpIndexMap(NWorldCoarse)
        fixedCoarse = util.boundarypIndexMap(NWorldCoarse)
        
        self.assertTrue(np.allclose(nodeFluxes[freeCoarse], 0))
            
        PCoarse = fem.assembleProlongationMatrix(NWorldCoarse, NCoarseElement)
        ROmega = PCoarse.T*(AFull*uFull - bFull)
        boundaryFlux = nodeflux.computeBoundaryFlux(world, ROmega)
        
        boundaryFluxRefx0 = -(1.*(xCoarse < 0+1e-7))*(2*np.pi*aCoef*np.cos(2*np.pi*aCoef*xCoarse)*np.sin(2*np.pi*bCoef*yCoarse))
        boundaryFluxRefx1 =  (1.*(xCoarse > 1-1e-7))*(2*np.pi*aCoef*np.cos(2*np.pi*aCoef*xCoarse)*np.sin(2*np.pi*bCoef*yCoarse))
        boundaryFluxRefy0 = -(1.*(yCoarse < 0+1e-7))*(2*np.pi*bCoef*np.sin(2*np.pi*aCoef*xCoarse)*np.cos(2*np.pi*bCoef*yCoarse))
        boundaryFluxRefy1 =  (1.*(yCoarse > 1-1e-7))*(2*np.pi*bCoef*np.sin(2*np.pi*aCoef*xCoarse)*np.cos(2*np.pi*bCoef*yCoarse))
        boundaryFluxRef = boundaryFluxRefx0 + boundaryFluxRefx1 + boundaryFluxRefy0 + boundaryFluxRefy1
        boundaryFluxRef[freeCoarse] = 0
        relativeDiscreteErrorNorm = np.linalg.norm((boundaryFlux - boundaryFluxRef)[fixedCoarse])/np.linalg.norm(boundaryFlux[fixedCoarse])
        self.assertTrue(np.isclose(relativeDiscreteErrorNorm, 0, atol=1e-2))

        MLocGetter = fem.localBoundaryMassMatrixGetter(NWorldCoarse)
        MBoundaryFull = fem.assemblePatchBoundaryMatrix(NWorldCoarse,
                                                        MLocGetter)
        self.assertTrue(np.isclose(np.max(np.abs(MBoundaryFull*boundaryFluxRef-nodeFluxes)), 0, atol=1e-2))
        
class computeMeanElementwiseQuantity_TestCase(unittest.TestCase):
    def test_computeMeanElementwiseQuantity_1d(self):
        NWorldCoarse = np.array([10])
        NCoarseElement = np.array([5])
        boundaryConditions = np.array([[0, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        boundaryFsValues = np.array([[-11, 101]])
        fsT = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19])

        nodeFs = nodeflux.computeMeanElementwiseQuantity(world, boundaryFsValues, fsT)

        self.assertTrue(np.allclose(nodeFs, [-5, 2, 4, 6, 8, 10, 12, 14, 16, 18, 60]))

        boundaryConditions = np.array([[1, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        boundaryFsValues = np.array([[np.nan, 101]])
        nodeFs = nodeflux.computeMeanElementwiseQuantity(world, boundaryFsValues, fsT)
        self.assertTrue(np.allclose(nodeFs, [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 60]))

        boundaryConditions = np.array([[0, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        boundaryFsValues = np.array([[0, 101]])
        nodeFs = nodeflux.computeMeanElementwiseQuantity(world, boundaryFsValues, fsT)
        self.assertTrue(np.allclose(nodeFs, [0.5, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]))

        boundaryConditions = np.array([[1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)
        boundaryFsValues = np.array([[222, 111]])
        nodeFs = nodeflux.computeMeanElementwiseQuantity(world, boundaryFsValues, fsT)
        self.assertTrue(np.allclose(nodeFs, [1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19]))

    def test_computeMeanElementwiseQuantity_2d(self):
        NWorldCoarse = np.array([3, 3])
        NCoarseElement = np.array([5, 5])
        boundaryConditions = np.array([[0, 0],
                                       [0, 0]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        boundaryFsValues = np.array([[0, 10],
                                     [20, 30]])
        fsT = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

        nodeFs = nodeflux.computeMeanElementwiseQuantity(world, boundaryFsValues, fsT)
        self.assertTrue(np.allclose(nodeFs, [12.5, 17.5, 22.5, 20,
                                             12.5, 30, 40, 27.5,
                                             27.5, 60, 70, 42.5,
                                             32.5, 52.5, 57.5, 40]))

        boundaryConditions = np.array([[0, 1],
                                       [1, 1]])
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        boundaryFsValues = np.array([[0, np.nan],
                                     [np.nan, np.nan]])
        fsT = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90])

        nodeFs = nodeflux.computeMeanElementwiseQuantity(world, boundaryFsValues, fsT)
        self.assertTrue(np.allclose(nodeFs, [5, 15, 25, 30,
                                             12.5, 30, 40, 45,
                                             27.5, 60, 70, 75,
                                             35, 75, 85, 90]))

class computeElementNetFlux_TestCase(unittest.TestCase):
    def test_computeElementNetFlux_1d(self):
        NWorldCoarse = np.array([4])
        NCoarseElement = np.array([5])
        boundaryConditions = np.array([[0, 0]])
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        boundaryFsValues = np.array([[0, 0]])
        nodeFluxT = np.array([[1, -2],
                              [2, -4],
                              [4,  1],
                              [-1, 0]])
        fsT = np.array([2, 2, 3, 1])

        # avg fs       = 1  2  2.5   2  0.5      net
        # avg fs*flux  = 1 -4                    -3
        #                   4  -10               -6
        #                       10   2           12 
        #                           -2    0      -2
        
        netFluxT = nodeflux.computeElementNetFlux(world, boundaryFsValues, fsT, nodeFluxT)
        self.assertTrue(np.allclose(netFluxT, [-3, -6, 12, -2]))

    def test_computeElementNetFlux_2d(self):
        NWorldCoarse = np.array([1, 1])
        NCoarseElement = np.array([5, 5])
        boundaryConditions = np.array([[0, 0],
                                       [1, 1]])
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        boundaryFsValues = np.array([[0, 20],
                                     [np.nan, np.nan]])
        nodeFluxT = np.array([[1, 2, 3, 4]])
        fsT = np.array([10])

        # (0+10)/2 = 5
        # (20+10)/2 = 15
        # 1*5 + 30 + 15 + 60 = 110
        
        netFluxT = nodeflux.computeElementNetFlux(world, boundaryFsValues, fsT, nodeFluxT)
        self.assertTrue(np.allclose(netFluxT, [110]))

if __name__ == '__main__':
    #import cProfile
    #command = """unittest.main()"""
    #cProfile.runctx( command, globals(), locals(), filename="test_pg.profile" )
    unittest.main()
