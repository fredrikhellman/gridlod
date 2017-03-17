import lod
import interp
from copy import deepcopy

world = None
coefficient = None
saddleSolver = None
k = None
IPatchGenerator = None
clearFineQuantities = None

def setupWorker(worldIn, coefficientIn, IPatchGeneratorIn, kIn, clearFineQuantitiesIn):
    global world, coefficient, saddleSolver, IPatchGenerator, k, clearFineQuantities

    world = worldIn
    coefficient = coefficientIn
    saddleSolver = lod.schurComplementSolver(world.NWorldCoarse*world.NCoarseElement)
    k = kIn
    if IPatchGeneratorIn is not None:
        IPatchGenerator = IPatchGeneratorIn
    else:
        IPatchGenerator = lambda i, N: interp.L2ProjectionPatchMatrix(i, N, world.NWorldCoarse,
                                                                      world.NCoarseElement,
                                                                      world.boundaryConditions)
    clearFineQuantities = clearFineQuantitiesIn

def computeElementCorrector(iElement):
    ecT = lod.elementCorrector(world, k, iElement, saddleSolver)
    coefficientPatch = coefficient.localize(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)
    IPatch = IPatchGenerator(ecT.iPatchWorldCoarse, ecT.NPatchCoarse)

    ecT.computeCorrectors(coefficientPatch, IPatch)
    ecT.computeCoarseQuantities()
    if clearFineQuantities:
        ecT.clearFineQuantities()
        
    return ecT
