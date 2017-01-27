import numpy as np
import scipy.sparse as sparse

import lod
import util


class PetrovGalerkinLOD:
    def __init__(self, world, k, IPatchGenerator, epsilonTol, printLevel=0):
        self.world = world
        NtCoarse = np.prod(world.NWorldCoarse)
        self.ecList = [None]*NtCoarse
        self.k = k
        self.IPatchGenerator = IPatchGenerator
        self.epsilonTol = epsilonTol
        self.printLevel = printLevel

    def updateCorrectors(self, coefficient, clearFineQuantities=True):
        world = self.world
        k = self.k
        IPatchGenerator = self.IPatchGenerator
        epsilonTol = self.epsilonTol
        
        NtCoarse = np.prod(world.NWorldCoarse)

        saddleSolver = lod.schurComplementSolver(world.NWorldCoarse*world.NCoarseElement)
        
        ecList = self.ecList
        for TInd in range(NtCoarse):
            if self.printLevel > 0:
                print str(TInd) + ' / ' + str(NtCoarse),
            iElement = util.convertpIndexToCoordinate(world.NWorldCoarse-1, TInd)
            if ecList[TInd] is not None  and  hasattr(coefficient, 'rCoarse'):
                ecT = ecList[TInd]
                coefficientPatch = coefficient.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                epsilonT = ecList[TInd].computeErrorIndicator(coefficientPatch.rCoarse)
            else:
                coefficientPatch = None
                epsilonT = np.inf
            
            if self.printLevel > 0:
                print 'epsilonT = ' + str(epsilonT)
                
            if epsilonT > epsilonTol:
                ecT = lod.elementCorrector(world, k, iElement, saddleSolver)

                if coefficientPatch is None:
                    coefficientPatch = coefficient.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                
                ecT.computeCorrectors(coefficientPatch, IPatch)
                ecT.computeCoarseQuantities()
                if clearFineQuantities:
                    ecT.clearFineQuantities()
                ecList[TInd] = ecT
                
    def clearCorrectors(self):
        NtCoarse = np.prod(self.world.NWorldCoarse)
        self.ecList = [None]*NtCoarse

    def assembleBasisCorrectors(self):
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
        
        return basisCorrectors
        
    def assembleMsStiffnessMatrix(self):
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
        
        return Kms

    def assembleStiffnessMatrix(self):
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
        
        return K
    
