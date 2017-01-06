import fem

class World:
    def __init__(self, NWorldCoarse, NCoarseElement):
        self.NWorldCoarse = NWorldCoarse
        self.NCoarseElement = NCoarseElement

    @property
    def localBasis(self):
        if not hasattr(self, '_localBasis'):
            self._localBasis = fem.localBasis(self.NCoarseElement)
        return self._localBasis
    
    @property
    def MLoc(self):
        if not hasattr(self, '_MLoc'):
            self._MLoc = fem.localMassMatrix(self.NWorldCoarse)
        return self._MLoc

    @property
    def ALoc(self):
        if not hasattr(self, '_ALoc'):
            self._ALoc = fem.localStiffnessMatrix(self.NWorldCoarse)
        return self._ALoc


