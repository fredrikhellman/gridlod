import numpy as np
import scipy.sparse as sparse
from copy import deepcopy

import lod
import util


class PetrovGalerkinLOD:
    def __init__(self, world, k, IPatchGenerator, epsilonTol, printLevel=0):
        self.world = world
        NtCoarse = np.prod(world.NWorldCoarse)
        self.k = k
        self.IPatchGenerator = IPatchGenerator
        self.epsilonTol = epsilonTol
        self.printLevel = printLevel

        self.epsilonList = None
        self.ageList = None
        self.ecList = None
        self.Kms = None
        self.K = None
        self.basisCorrectors = None
        self.coefficient = None
        
    def updateCorrectors(self, coefficient, clearFineQuantities=True):
        world = self.world
        k = self.k
        IPatchGenerator = self.IPatchGenerator
        epsilonTol = self.epsilonTol
        
        NtCoarse = np.prod(world.NWorldCoarse)

        saddleSolver = lod.schurComplementSolver(world.NWorldCoarse*world.NCoarseElement)

        self.coefficient = deepcopy(coefficient)
        
        # Reset all caches
        self.Kms = None
        self.K = None
        self.basisCorrectors = None
        
        if self.ecList is None:
            self.ecList = [None]*NtCoarse

        if self.ageList is None:
            self.ageList = [-1]*NtCoarse

        if self.epsilonList is None:
            self.epsilonList = [np.nan]*NtCoarse

        ecList = self.ecList
        ageList = self.ageList
        epsilonList = self.epsilonList
        for TInd in range(NtCoarse):
            if self.printLevel > 0:
                print str(TInd) + ' / ' + str(NtCoarse),

            ageList[TInd] += 1
            iElement = util.convertpIndexToCoordinate(world.NWorldCoarse-1, TInd)
            if ecList[TInd] is not None  and  hasattr(coefficient, 'rCoarse'):
                ecT = ecList[TInd]
                coefficientPatch = coefficient.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                epsilonT = ecList[TInd].computeErrorIndicator(coefficientPatch.rCoarse)
            else:
                coefficientPatch = None
                epsilonT = np.inf

            epsilonList[TInd] = epsilonT
            
            if self.printLevel > 0:
                print 'epsilonT = ' + str(epsilonT), 
                
            if epsilonT > epsilonTol:
                if self.printLevel > 0:
                    print 'C'
                ecT = lod.elementCorrector(world, k, iElement, saddleSolver)

                if coefficientPatch is None:
                    coefficientPatch = coefficient.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                
                ecT.computeCorrectors(coefficientPatch, IPatch)
                ecT.computeCoarseQuantities()
                if clearFineQuantities:
                    ecT.clearFineQuantities()
                ecList[TInd] = ecT
                ageList[TInd] = 0
            else:
                if self.printLevel > 0:
                    print 'N'

    def clearCorrectors(self):
        NtCoarse = np.prod(self.world.NWorldCoarse)
        self.ecList = None
        self.coefficient = None

    def computeCorrection(self, ARhsFull=None, MRhsFull=None):
        assert(self.ecList is not None)
        assert(self.coefficient is not None)

        world = self.world
        NCoarseElement = world.NCoarseElement
        NWorldCoarse = world.NWorldCoarse
        NWorldFine = NWorldCoarse*NCoarseElement

        NpFine = np.prod(NWorldFine+1)
        
        coefficient = self.coefficient
        IPatchGenerator = self.IPatchGenerator

        localBasis = world.localBasis
        
        TpIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NWorldFine)
        TpStartIndices = util.pIndexMap(NWorldCoarse-1, NWorldFine, NCoarseElement)
        
        uFine = np.zeros(NpFine)
        
        NtCoarse = np.prod(world.NWorldCoarse)
        for TInd in range(NtCoarse):
            if self.printLevel > 0:
                print str(TInd) + ' / ' + str(NtCoarse)
                
            ecT = self.ecList[TInd]
            
            coefficientPatch = coefficient.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
            IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

            if ARhsFull is not None:
                ARhsList = [ARhsFull[TpStartIndices[TInd] + TpIndexMap]]
            else:
                ARhsList = None
                
            if MRhsFull is not None:
                MRhsList = [MRhsFull[TpStartIndices[TInd] + TpIndexMap]]
            else:
                MRhsList = None
                
            correctorT = ecT.computeElementCorrector(coefficientPatch, IPatch, ARhsList, MRhsList)[0]
            
            NPatchFine = ecT.NPatchCoarse*NCoarseElement
            iPatchWorldFine = ecT.iPatchWorldCoarse*NCoarseElement
            patchpIndexMap = util.lowerLeftpIndexMap(NPatchFine, NWorldFine)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldFine, iPatchWorldFine)

            uFine[patchpStartIndex + patchpIndexMap] += correctorT

        return uFine
    
    def assembleBasisCorrectors(self):
        if self.basisCorrectors is not None:
            return self.basisCorrectors

        assert(self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse
        NCoarseElement = world.NCoarseElement
        NWorldFine = NWorldCoarse*NCoarseElement
        
        NtCoarse = np.prod(NWorldCoarse)
        NpCoarse = np.prod(NWorldCoarse+1)
        NpFine = np.prod(NWorldFine+1)
        
        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)
        
        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert(ecT is not None)
            assert(hasattr(ecT, 'fsi'))

            NPatchFine = ecT.NPatchCoarse*NCoarseElement
            iPatchWorldFine = ecT.iPatchWorldCoarse*NCoarseElement
            
            patchpIndexMap = util.lowerLeftpIndexMap(NPatchFine, NWorldFine)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldFine, iPatchWorldFine)
            
            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = patchpStartIndex + patchpIndexMap
            dataT = np.hstack(ecT.fsi.correctorsList)

            cols.extend(np.repeat(colsT, np.size(rowsT)))
            rows.extend(np.tile(rowsT, np.size(colsT)))
            data.extend(dataT)

        basisCorrectors = sparse.csc_matrix((data, (rows, cols)), shape=(NpFine, NpCoarse))

        self.basisCorrectors = basisCorrectors
        return basisCorrectors
        
    def assembleMsStiffnessMatrix(self):
        if self.Kms is not None:
            return self.Kms

        assert(self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse
        
        NtCoarse = np.prod(world.NWorldCoarse)
        NpCoarse = np.prod(world.NWorldCoarse+1)
        
        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert(ecT is not None)

            NPatchCoarse = ecT.NPatchCoarse

            patchpIndexMap = util.lowerLeftpIndexMap(NPatchCoarse, NWorldCoarse)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldCoarse, ecT.iPatchWorldCoarse)
            
            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = patchpStartIndex + patchpIndexMap
            dataT = ecT.csi.Kmsij.flatten()

            cols.extend(np.tile(colsT, np.size(rowsT)))
            rows.extend(np.repeat(rowsT, np.size(colsT)))
            data.extend(dataT)

        Kms = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

        self.Kms = Kms
        return Kms

    def assembleStiffnessMatrix(self):
        if self.K is not None:
            return self.K

        assert(self.ecList is not None)

        world = self.world
        NWorldCoarse = world.NWorldCoarse
        
        NtCoarse = np.prod(world.NWorldCoarse)
        NpCoarse = np.prod(world.NWorldCoarse+1)
        
        TpIndexMap = util.lowerLeftpIndexMap(np.ones_like(NWorldCoarse), NWorldCoarse)
        TpStartIndices = util.lowerLeftpIndexMap(NWorldCoarse-1, NWorldCoarse)

        cols = []
        rows = []
        data = []
        ecList = self.ecList
        for TInd in range(NtCoarse):
            ecT = ecList[TInd]
            assert(ecT is not None)

            NPatchCoarse = ecT.NPatchCoarse

            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = TpStartIndices[TInd] + TpIndexMap
            dataT = ecT.csi.Kij.flatten()

            cols.extend(np.tile(colsT, np.size(rowsT)))
            rows.extend(np.repeat(rowsT, np.size(colsT)))
            data.extend(dataT)

        K = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))

        self.K = K
        return K
    
