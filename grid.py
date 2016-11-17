import numpy as np

from util import *

class grid:
    '''
    Grid dD class for a two level uniform grid with lexicographical
    node ordering. Constructed to handle huge meshes. Data not to be 
    stored explicitly.
    '''

    def __init__(self, NWorldCoarse, NCoarseElement=None):
        self.d = np.size(NWorldCoarse)
        self.NWorldCoarse = np.array(NWorldCoarse).astype('int64')
        if NCoarseElement is None:
            self.NCoarseElement = np.ones(self.d, dtype='int64')
        else:
            self.NCoarseElement = np.array(NCoarseElement).astype('int64')
        self.NWorldFine = self.NWorldCoarse*self.NCoarseElement
        
class gridPatch:
    def __init__(self, g, iPatchCoarse=None, NPatchCoarse=None):
        self.g = g
        if iPatchCoarse is None:
            self.iPatchCoarse = np.zeros(g.d, dtype='int64')
        else:
            self.iPatch = np.array(iPatch).astype('int64')

        if NPatchCoarse is None:
            self.NPatchCoarse = g.NWorldCoarse-self.iPatchCoarse
        else:
            self.NPatchCoarse = np.array(NPatchCoarse).astype('int64')

        self.NPatchFine = g.NCoarseElement*self.NPatchCoarse
        self.iPatchFine = g.NCoarseElement*self.iPatchCoarse
            
    def pCoarse(self):
        g = self.g
        return pCoordinates(g.NWorldCoarse, self.iPatchCoarse, self.NPatchCoarse)

    def pFine(self):
        g = self.g
        return pCoordinates(g.NWorldFine, self.iPatchFine, self.NPatchFine)
