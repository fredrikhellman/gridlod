# This file contains the code that is specific for the first paper by
# Målqvist and Hellman, i.e. in the settings of time-dependent
# problems with lagging coefficients and flux coarse quantities.
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

from . import fem
from . import lod
from . import util
from . import linalg
from . import coef
from . import transport

class CoarseScaleInformationFlux:
    def __init__(self, correctorFluxTF, basisFluxTF, rCoarse=None):
        self.rCoarse = rCoarse
        self.correctorFluxTF = correctorFluxTF
        self.basisFluxTF = basisFluxTF

class CoarseBasisElementCorrectorFlux(lod.CoarseBasisElementCorrector):
    def __init__(self):
        super().__init__()
        self.csiFlux = None
    
    def computeCoarseQuantitiesFlux(self):
        assert(self.fsi is not None)

        world = self.world
        NCoarseElement = world.NCoarseElement
        NPatchCoarse = self.NPatchCoarse
        NPatchFine = NPatchCoarse*NCoarseElement

        correctorsList = self.fsi.correctorsList
        aPatch = self.fsi.coefficient.aFine
        QPatch = np.column_stack(correctorsList)
        
        localBasis = world.localBasis

        TPrimeFinepStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine, NCoarseElement)
        TPrimeFinepIndexMap = util.lowerLeftpIndexMap(NCoarseElement, NPatchFine)

        TInd = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, self.iElementPatchCoarse)
        
        if aPatch.ndim == 1:
            # Face flux computations are only possible for scalar coefficients
            # Compute coarse element face flux for basis and Q (not yet implemented for R)
            correctorFluxTF = transport.computeHarmonicMeanFaceFlux(world.NWorldCoarse,
                                                                    NPatchCoarse,
                                                                    NCoarseElement, aPatch, QPatch)

            # Need to compute over at least one fine element over the
            # boundary of the main element to make the harmonic average
            # right.  Note: We don't do this for correctorFluxTF, beause
            # we do not have access to a outside the patch...
            localBasisExtended = np.zeros_like(QPatch)
            localBasisExtended[TPrimeFinepStartIndices[TInd] + TPrimeFinepIndexMap,:] = localBasis
            basisFluxTF = transport.computeHarmonicMeanFaceFlux(world.NWorldCoarse,
                                                                NPatchCoarse,
                                                                NCoarseElement, aPatch, localBasisExtended)[:,TInd,:]
        if isinstance(self.fsi.coefficient, coef.coefficientCoarseFactorAbstract):
            rCoarse = self.fsi.coefficient.rCoarse
        else:
            rCoarse = None
        self.csiFlux = CoarseScaleInformationFlux(correctorFluxTF, basisFluxTF, rCoarse)

    def computeCoarseQuantities(self):
        super().computeCoarseQuantities()
        self.computeCoarseQuantitiesFlux()

    def computeErrorIndicatorFineWithLagging(self, a, aTilde):
        assert(self.csi is not None)

        assert(a.ndim == 1) # Matrix-valued A not supported in thus function yet
        
        world = self.world
        NPatchCoarse = self.NPatchCoarse
        NCoarseElement = world.NCoarseElement
        NPatchFine = NPatchCoarse*NCoarseElement
        iElementPatchCoarse = self.iElementPatchCoarse

        elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, iElementPatchCoarse)
        
        TPrimeFinetStartIndices = util.pIndexMap(NPatchCoarse-1, NPatchFine-1, NCoarseElement)
        TPrimeFinetIndexMap = util.lowerLeftpIndexMap(NCoarseElement-1, NPatchFine-1)

        muTPrime = self.csi.muTPrime

        TPrimeIndices = np.add.outer(TPrimeFinetStartIndices, TPrimeFinetIndexMap)
        aTPrime = a[TPrimeIndices]
        aTildeTPrime = aTilde[TPrimeIndices]
        
        deltaMaxNormTPrime = np.max(np.abs((aTPrime - aTildeTPrime)/np.sqrt(aTPrime*aTildeTPrime)), axis=1)
        theOtherUnnamedFactorTPrime = np.max(np.abs(aTPrime[elementCoarseIndex]/aTildeTPrime[elementCoarseIndex]))

        epsilonTSquare = theOtherUnnamedFactorTPrime * \
                         np.sum((deltaMaxNormTPrime**2)*muTPrime)

        return np.sqrt(epsilonTSquare)
        
    def computeCoarseErrorIndicatorFlux(self, rCoarseNew):
        assert(self.csi is not None)
        assert(self.csiFlux is not None)
        assert(self.csiFlux.rCoarse is not None)
        
        world = self.world
        NPatchCoarse = self.NPatchCoarse
        NCoarseElement = world.NCoarseElement
        iElementPatchCoarse = self.iElementPatchCoarse

        elementCoarseIndex = util.convertpCoordIndexToLinearIndex(NPatchCoarse-1, iElementPatchCoarse)
        
        rCoarse = self.csiFlux.rCoarse
        muTPrime = self.csi.muTPrime
        deltaMaxNormTPrime = np.abs((rCoarseNew - rCoarse)/np.sqrt(rCoarseNew*rCoarse))
        
        epsilonTSquare = rCoarseNew[elementCoarseIndex]/rCoarse[elementCoarseIndex] * \
                         np.sum((deltaMaxNormTPrime**2)*muTPrime)

        return np.sqrt(epsilonTSquare)
