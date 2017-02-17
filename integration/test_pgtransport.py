import unittest
import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
from itertools import count
import os
import sys

import matplotlib.pyplot as plt
from pyevtk.hl import imageToVTK 

from gridlod import pg, transport, nodeflux, interp, coef, util, fem, world, linalg, femsolver
from gridlod.world import World

def drawCoefficient(N, a):
    aCube = np.log10(a.reshape(N, order='F'))
    aCube = np.ascontiguousarray(aCube.T)
    plt.clf()
    
    cmap = plt.cm.viridis
            
    plt.imshow(aCube,
               origin='lower',
               interpolation='none',
               cmap=cmap)
    plt.colorbar()
    
def drawSaturation(N, s):
    sCube = s.reshape(N, order='F')
    sCube = np.ascontiguousarray(sCube.T)
    plt.clf()
    
    cmap = plt.cm.viridis
    cmap.set_under('b')
    cmap.set_over('r')
            
    plt.imshow(sCube,
               origin='lower',
               interpolation='none',
               cmap=cmap,
               clim=[0, 1],
               extent=[0, 1, 0, 1])
    plt.colorbar()

def projectSaturation(NWorldCoarse, NCoarseElement, sFine):
    NWorldFine = NWorldCoarse*NCoarseElement
    
    TIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NWorldFine-1)
    TStartIndices = util.pIndexMap(NWorldCoarse-1, NWorldFine-1, NCoarseElement)

    NtCoarse = np.prod(NWorldCoarse)
    sCoarse = np.zeros(NtCoarse)
    for TInd in range(NtCoarse):
        sCoarse[TInd] = np.mean(sFine[TStartIndices[TInd] + TIndexMap])

    return sCoarse

class IntegrationPGTransport_TestCase(unittest.TestCase):
    def test_pgtransport(self):
        NWorldCoarse = np.array([16, 16])
        NCoarseElement = np.array([8, 8])
        NWorldFine = NWorldCoarse*NCoarseElement
        boundaryConditions = np.array([[1, 1],
                                       [0, 0]])

        d = 2
        
        world = World(NWorldCoarse, NCoarseElement, boundaryConditions)

        # Load coefficient
        aLogBase = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/data/randomfield64x64')
        #aBase = np.loadtxt(os.path.dirname(os.path.realpath(__file__)) + '/data/upperness_x.txt')
        #aBase = aBase[:60*220]
        #aLogBase = np.random.rand(128*128)
        aBase = np.exp(3*aLogBase)

        drawCoefficient(NWorldFine, aBase)
        plt.title('a')
        
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
        gFine = 1-yFine
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

            if True:
                avgFluxTF = transport.computeHarmonicMeanFaceFlux(fluxWorld.NWorldCoarse, fluxWorld.NWorldCoarse, NFluxElement, aBase, uFull)
                avgFluxTF = transport.computeAverageFaceFlux(fluxWorld.NWorldCoarse, avgFluxTF)
                conservativeFluxTF = transport.computeConservativeFlux(fluxWorld, avgFluxTF)

            if False:
                avgFluxTF = transport.computeElementFaceFlux(fluxWorld.NWorldCoarse, fluxWorld.NWorldCoarse, NFluxElement, aBase, uFull)
                avgFluxTF = transport.computeAverageFaceFlux(fluxWorld.NWorldCoarse, avgFluxTF)
            
            return avgFluxTF, conservativeFluxTF

        avgFluxTF, conservativeFluxTF = computeAvgVelocitiesTF(NCoarseElement)
        avgFluxtf, conservativeFluxtf = computeAvgVelocitiesTF(np.ones_like(NCoarseElement))

        def fractionalFlow(s):
            return s**3

        boundarys = np.array([[0, 0], [1, 0]])

        sT = np.zeros(world.NtCoarse)
        st = np.zeros(world.NtFine)

        nTime = 1e5
        endTime = 1
        dTime = endTime/float(nTime)
        volt = np.prod(1./world.NWorldFine)
        volT = np.prod(1./world.NWorldCoarse)

        plt.figure()
        h1 = plt.gcf().number
        plt.figure()
        h2 = plt.gcf().number
        plt.figure()
        h3 = plt.gcf().number
        
        for timeStep in np.arange(nTime):
            def computeElementNetFluxT(NFluxElement, avgFluxTF, sT):
                fluxWorld = World(NWorldFine/NFluxElement, NFluxElement, boundaryConditions)
                netFluxT = transport.computeElementNetFlux(fluxWorld, avgFluxTF, sT, boundarys, fractionalFlow)
                return netFluxT

            netFluxT = computeElementNetFluxT(NCoarseElement, conservativeFluxTF, sT)
            netFluxt = computeElementNetFluxT(np.ones_like(NCoarseElement), conservativeFluxtf, st)

            sT = sT + dTime/volT*netFluxT
            #sT[sT > 1] = 1.
            #sT[sT < 0] = 0.
                
            st = st + dTime/volt*netFluxt
            #st[st > 1] = 1.
            #st[st < 0] = 0.


            if timeStep%1000 == 0 or timeStep == nTime-1:
                plt.figure(h1)
                drawSaturation(NWorldCoarse, sT)
                plt.title('sT')
                plt.gcf().canvas.draw()
                
                plt.figure(h2)
                drawSaturation(NWorldFine, st)
                plt.title('st')
                plt.gcf().canvas.draw()

                plt.figure(h3)
                stProjected = projectSaturation(NWorldCoarse, NCoarseElement, st)
                drawSaturation(NWorldCoarse, stProjected)
                plt.title('st projected')
                plt.gcf().canvas.draw()

                print np.sqrt(np.mean((stProjected-sT)**2))
                
                plt.pause(0.0001)
        plt.show()
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

