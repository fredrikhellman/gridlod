import numpy as np

import util

class coefficientAbstract:
    @property
    def aFine(self):
        raise(NotImplementedError('Abstract class'))

class coefficientCoarseFactorAbstract:
    @property
    def rCoarse(self):
        raise(NotImplementedError('Abstract class'))
    
    
class coefficientFine(coefficientAbstract):
    def __init__(self, NPatchCoarse, NCoarseElement, aFine):
        self.NPatchCoarse = np.array(NPatchCoarse)
        self.NCoarseElement = NCoarseElement
        self._aFine = aFine
        assert(np.size(aFine) == np.prod(NPatchCoarse*NCoarseElement))

    def localize(self, iSubPatchCoarse, NSubPatchCoarse):
        NPatchCoarse = self.NPatchCoarse
        NCoarseElement = self.NCoarseElement
        NPatchFine = NPatchCoarse*NCoarseElement
        NSubPatchFine = NSubPatchCoarse*NCoarseElement
        iSubPatchFine = iSubPatchCoarse*NCoarseElement

        # a
        coarsetIndexMap = util.lowerLeftpIndexMap(NSubPatchFine-1, NPatchFine-1)
        coarsetStartIndex = util.convertpCoordinateToIndex(NPatchFine-1, iSubPatchFine)
        aFineLocalized = self._aFine[coarsetStartIndex + coarsetIndexMap]

        localizedCoefficient = coefficientFine(NSubPatchCoarse, NCoarseElement, aFineLocalized)
        return localizedCoefficient

    @property
    def aFine(self):
        return self._aFine

class coefficientCoarseFactor(coefficientAbstract, coefficientCoarseFactorAbstract):
    def __init__(self, NPatchCoarse, NCoarseElement, aBase, rCoarse):
        self.NPatchCoarse = np.array(NPatchCoarse)
        self.NCoarseElement = NCoarseElement
        self._aBase = aBase
        self._rCoarse = rCoarse

    def precompute(self):
        if hasattr(self, '_aFine'):
            return
        
        NPatchCoarse = self.NPatchCoarse
        NCoarseElement = self.NCoarseElement
        NPatchFine = NPatchCoarse*NCoarseElement
        
        coarsetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
        coarsetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

        coarsetIndexTensor = np.add.outer(coarsetIndexMap, coarsetStartIndices)
        self._aFine = np.array(self._aBase)
        self._aFine[coarsetIndexTensor] *= self._rCoarse
        del self._aBase

    def localize(self, iSubPatchCoarse, NSubPatchCoarse):
        # Either ._aBase or ._aFine exists, not both
        if not hasattr(self, '_aFine'):
            a = self._aBase
        else:
            a = self._aFine

        NPatchCoarse = self.NPatchCoarse
        NCoarseElement = self.NCoarseElement
        NPatchFine = NPatchCoarse*NCoarseElement
        NSubPatchFine = NSubPatchCoarse*NCoarseElement
        iSubPatchFine = iSubPatchCoarse*NCoarseElement

        # rCoarse
        coarseTIndexMap = util.lowerLeftpIndexMap(NSubPatchCoarse-1, NPatchCoarse-1)
        coarseTStartIndex = util.convertpCoordinateToIndex(NPatchCoarse-1, iSubPatchCoarse)
        rCoarseLocalized = self._rCoarse[coarseTStartIndex + coarseTIndexMap]

        # a
        coarsetIndexMap = util.lowerLeftpIndexMap(NSubPatchFine-1, NPatchFine-1)
        coarsetStartIndex = util.convertpCoordinateToIndex(NPatchFine-1, iSubPatchFine)
        aLocalized = a[coarsetStartIndex + coarsetIndexMap]

        localizedCoefficient = coefficientCoarseFactor(NSubPatchCoarse, NCoarseElement, None, rCoarseLocalized)
        if not hasattr(self, '_aFine'):
            localizedCoefficient._aBase = aLocalized
        else:
            localizedCoefficient._aFine = aLocalized
            del localizedCoefficient._aBase

        return localizedCoefficient
            
    @property
    def aFine(self):
        if not hasattr(self, '_aFine'):
            self.precompute()
        return self._aFine
        
    @property
    def rCoarse(self):
        return self._rCoarse
