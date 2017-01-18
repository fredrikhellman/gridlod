import numpy as np
import scipy.sparse as sparse

import lod
import util


class PetrovGalerkinLOD:
    def __init__(self, world, k, IPatchGenerator, epsilonTol):
        self.world = world
        NtCoarse = np.prod(world.NWorldCoarse)
        self.ecList = [None]*NtCoarse
        self.k = k
        self.IPatchGenerator = IPatchGenerator
        self.epsilonTol = epsilonTol

    def updateCorrectors(self, coefficient):
        world = self.world
        k = self.k
        IPatchGenerator = self.IPatchGenerator
        epsilonTol = self.epsilonTol
        
        NtCoarse = np.prod(world.NWorldCoarse)

        ecList = self.ecList
        for TInd in range(NtCoarse):
            iElement = util.convertpIndexToCoordinate(world.NWorldCoarse-1, TInd)
            if ecList[TInd] is not None  and  hasattr(coefficient, 'rCoarse'):
                epsilonT = ecList[TInd].computeError(coefficient.rCoarse)
            else:
                epsilonT = np.inf

            if epsilonT > epsilonTol:
                ecT = lod.elementCorrector(world, k, iElement)

                coefficientPatch = coefficient.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
                
                ecT.computeCorrectors(coefficientPatch, IPatch)
                ecT.computeCoarseQuantities()
                ecT.clearFineQuantities()
                ecList[TInd] = ecT
                
    def clearCorrectors(self):
        NtCoarse = np.prod(self.world.NWorldCoarse)
        self.ecList = [None]*NtCoarse
        
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

            patchpIndexMap = util.lowerLeftpIndexMap(NPatchCoarse, NWorldCoarse)
            patchpStartIndex = util.convertpCoordinateToIndex(NWorldCoarse, ecT.iPatchWorldCoarse)
            
            colsT = TpStartIndices[TInd] + TpIndexMap
            rowsT = patchpStartIndex + patchpIndexMap
            dataT = ecT.csi.Kij.flatten()

            cols.extend(np.repeat(colsT, np.size(rowsT)))
            rows.extend(np.tile(rowsT, np.size(colsT)))
            data.extend(dataT)

        K = sparse.csc_matrix((data, (rows, cols)), shape=(NpCoarse, NpCoarse))
        
        return K
