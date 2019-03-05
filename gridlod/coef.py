import numpy as np

from . import util

def localizeCoefficient(patch, aFine):
    iPatchWorldCoarse = patch.iPatchWorldCoarse
    NPatchCoarse = patch.NPatchCoarse
    NCoarseElement = patch.world.NCoarseElement
    NPatchFine = NPatchCoarse*NCoarseElement
    iPatchWorldFine = iPatchWorldCoarse*NCoarseElement
    NWorldFine = patch.world.NWorldFine
    
    # a
    coarsetIndexMap = util.lowerLeftpIndexMap(NPatchFine-1, NWorldFine-1)
    coarsetStartIndex = util.convertpCoordIndexToLinearIndex(NWorldFine-1, iPatchWorldFine)
    aFineLocalized = aFine[coarsetStartIndex + coarsetIndexMap]
    return aFineLocalized


