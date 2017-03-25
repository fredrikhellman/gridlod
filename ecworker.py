import lod
import interp
import coef

from copy import deepcopy

world = None
coefficient = None
saddleSolver = None
k = None
IPatchGenerator = None
clearFineQuantities = None

# for rCoarse
aBase = None

# for aLagging
aLagging = None
aTrue = None

def clearWorker():
    global world, coefficient, saddleSolver, IPatchGenerator, k, clearFineQuantities, aBase, aLagging, aTrue
    world = None
    coefficient = None
    saddleSolver = None
    k = None
    IPatchGenerator = None
    clearFineQuantities = None
    aBase = None

    aLagging = None
    aTrue = None
    
def hasaBase():
    return aBase is not None

def sendar(aBaseIn, rCoarseIn):
    global aBase, coefficient
    if aBaseIn is not None:
        aBase = aBaseIn
    coefficient = coef.coefficientCoarseFactor(world.NWorldCoarse, world.NCoarseElement, aBase, rCoarseIn)

def sendas(aTrueIn, aLaggingIn):
    global aTrue, aLagging, coefficient
    aTrue = aTrueIn
    aLagging = aLaggingIn
    coefficient = coef.coefficientFineWithLagging(world.NWorldCoarse, world.NCoarseElement, aTrue, aLagging)
    
def setupWorker(worldIn, coefficientIn, IPatchGeneratorIn, kIn, clearFineQuantitiesIn):
    global world, coefficient, saddleSolver, IPatchGenerator, k, clearFineQuantities

    world = worldIn
    if coefficientIn is not None:
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
