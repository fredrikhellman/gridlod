import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
from itertools import count
import os
import sys

from pyevtk.hl import imageToVTK 

from gridlod import pg, transport, interp, coef, util, fem, world, linalg, femsolver
from gridlod.world import World

class IntegrationPGTransport_TestCase(unittest.TestCase):
    def test_pgtransport(self):
        NWorldCoarse = np.array([8, 8])
        NCoarseElement = np.array([8, 8])
        NWorldFine = NWorldCoarse*NCoarseElement
        boundaryConditions = np.array([[0, 0],
                                       [1, 1]])

        d = 2
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        # Load coefficient
        aLogBase = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/data/randomfield64x64')
        aLogBase = np.random.rand(128*128)
        aBase = np.exp(5*aLogBase[::4])

        # Compute coordinates
        coordsFine = util.pCoordinates(NWorldFine)
        xFine = coordsFine[:,0]
        yFine = coordsFine[:,1]
        
        # Compute fixed and free dofs
        allDofs = np.arange(world.NpFine)
        fixed = util.boundarypIndexMap(NWorldFine, boundaryConditions==0)
        free = np.setdiff1d(allDofs, fixed)

        # Compute matrices
        AFull = fem.assemblePatchMatrix(NWorldFine, world.ALocFine, aBase)
        A = AFull[free][:,free]

        # Compute rhs
        gFine = 1-xFine
        bFull = -AFull*gFine
        b = bFull[free]

        # Solve fine system
        u0Free = sparse.linalg.spsolve(A, b)
        u0Full = np.zeros(world.NpFine)
        u0Full[free] = u0Free

        uFull = u0Full + gFine

        ## First, compute flux on fine mesh
        def computeAvgVelocitiesTF(NFluxElement):
            fluxWorld = World(NWorldFine/NFluxElement, NFluxElement, boundaryConditions)

            avgVelocityTF = transport.computeElementFaceVelocity(fluxWorld.NWorldCoarse, NFluxElement, aBase, uFull)
            avgVelocityTF = transport.computeAverageFaceVelocity(fluxWorld.NWorldCoarse, avgVelocityTF)
            
            return avgVelocityTF

        avgVelocityTF = computeAvgVelocitiesTF(NCoarseElement)
        avgVelocitytf = computeAvgVelocitiesTF(np.ones_like(NCoarseElement))

        import matplotlib.pyplot as plt
        
        def fractionalFlow(s):
            return s**3

        boundarys = np.array([[1, 0], [0, 0]])

        sT = np.zeros(world.NtCoarse)
        st = np.zeros(world.NtFine)

        nTime = 1e5
        endTime = 1
        dTime = endTime/float(nTime)
        volt = np.prod(1./world.NWorldFine)
        volT = np.prod(1./world.NWorldCoarse)

        h1 = None
        h2 = None
        for timeStep in np.arange(nTime):
            def computeElementNetFluxT(NFluxElement, avgVelocityTF, sT):
                fluxWorld = World(NWorldFine/NFluxElement, NFluxElement, boundaryConditions)
                netFluxT = transport.computeElementNetFlux(fluxWorld, avgVelocityTF, sT, boundarys, fractionalFlow)
                return netFluxT

            netFluxT = computeElementNetFluxT(NCoarseElement, avgVelocityTF, sT)
            netFluxt = computeElementNetFluxT(np.ones_like(NCoarseElement), avgVelocitytf, st)

            sT = sT + dTime/volT*netFluxT
            sT[sT > 1] = 1.
            sT[sT < 0] = 0.
                
            st = st + dTime/volt*netFluxt
            st[st > 1] = 1.
            st[st < 0] = 0.

            print np.min(sT), np.max(sT)

            if timeStep%1000 == 0:
                sTCube = sT.reshape(NWorldCoarse)
                stCube = st.reshape(NWorldFine)
                #plt.imshow(sTCube, origin='lower')
                plt.figure(h1)
                plt.clf()
                plt.imshow(np.log10(np.abs(sTCube)+1e-8),
                           origin='lower',
                           interpolation='none',
                           cmap=plt.cm.viridis,
                           clim=[-8, 0])
                plt.colorbar()
                if h1 is None:
                    h1 = plt.gcf().number
                    plt.gcf().show()
                plt.gcf().canvas.draw()
                
                plt.figure(h2)
                plt.clf()
                plt.imshow(np.log10(np.abs(stCube)+1e-8),
                           origin='lower',
                           interpolation='none',
                           cmap=plt.cm.viridis,
                           clim=[-8, 0])
                plt.colorbar()
                if h2 is None:
                    h2 = plt.gcf().number
                    plt.gcf().show()
                plt.gcf().canvas.draw()
                plt.pause(0.0001)
            
'''            
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
'''

if __name__ == '__main__':
    #import cProfile
    #command = """unittest.main()"""
    #cProfile.runctx( command, globals(), locals(), filename="profile" )
    unittest.main()

