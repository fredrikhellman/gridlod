from . import util
import numpy as np

def _computeCoordinateIndexParts(N, coordinates):
    assert(np.all(coordinates <= 1.0) and np.all(coordinates >= 0.0))
    
    eps = np.finfo(np.float64).eps
    
    # Find what index coordinates the coordinates are in
    coordIndicesDecimal = coordinates.astype(np.float64)*N
    coordIndicesDecimal *= (1.-eps) # This guarantees we can call with coordinate 1.0
    
    # The integer part tells in what element to perform the interpolation
    # The fractional part gives the interpolation weights within the element
    coordIndicesIpart = np.floor(coordIndicesDecimal).astype(np.int64)
    coordIndicesFpart = coordIndicesDecimal - coordIndicesIpart

    return coordIndicesIpart, coordIndicesFpart

def evaluateDQ0(N, dq0, coordinates):
    """Compute a discontinuous piecewise constant (DQ0) function on an
       N-grid at the given coordinates. The function can be
       tensor-valued.

    """
    assert(coordinates.ndim == 2)
    assert(dq0.shape[0] == np.prod(N))
    
    coordIndices, _ = _computeCoordinateIndexParts(N, coordinates)
    
    # Convert to linear indices
    linearIndices = util.convertpCoordIndexToLinearIndex(N-1, coordIndices)

    # Lookup function values
    evaluatedDQ0 = dq0[linearIndices,...]
    
    return evaluatedDQ0

def evaluateCQ1(N, cq1, coordinates):
    """Compute a continuous piecewise d-linear (CQ1) function on an N-grid
       at the given coordinates. The function can be tensor-valued.

    d-linear interpolation is used, i.e. linear in 1d, bilinear in 2d
    etc.

    """

    assert(coordinates.ndim == 2)
    assert(cq1.shape[0] == np.prod(N+1))

    d = N.size
    
    coordIndicesIpart, coordIndicesFpart = _computeCoordinateIndexParts(N, coordinates)

    # Compute interpolation weights
    interpolationWeights = np.zeros((coordinates.shape[0], 2**d))
    for localNode in range(2**d):
        localNodeBinaryString = bin(localNode)[2:].zfill(d)[::-1]
        localNodeBitmap = np.array([0 if bit == '0' else 1 for bit in localNodeBinaryString])

        # This computes the value of the basis function at node
        # localNode for all points. It is a product of either x_k or
        # 1-x_k for each dimension k, depending on the node. The
        # bitmap tells us which.

        interpolationWeights[:, localNode] = np.prod(
            (1.0 - localNodeBitmap) + (2.0*localNodeBitmap - 1.0)*coordIndicesFpart,
            axis=1)

    # Compute the global indices of all the local nodes
    coordIndicesLinear = util.convertpCoordIndexToLinearIndex(N, coordIndicesIpart)
    interpolationValuesIndices = np.add.outer(coordIndicesLinear, util.elementpIndexMap(N))

    # Compute the values of the relevant nodes
    interpolationValues = cq1[interpolationValuesIndices,...]

    # Perform the interpolation sum of weight-value products
    evaluatedCQ1 = np.einsum('pi,pi...->p...', interpolationWeights, interpolationValues)
    
    return evaluatedCQ1

def evaluateCQ1D(N, cq1, coordinates):
    """Compute the derivative of a continuous piecewise d-linear (cq1)
       function on an N-grid at the given coordinates. The function
       can be tensor-valued.

    """

    assert(coordinates.ndim == 2)
    assert(cq1.shape[0] == np.prod(N+1))

    d = N.size
    
    coordIndicesIpart, coordIndicesFpart = _computeCoordinateIndexParts(N, coordinates)

    # Compute interpolation weights
    interpolationWeights = np.zeros((coordinates.shape[0], 2**d, d))
    for localNode in range(2**d):
        localNodeBinaryString = bin(localNode)[2:].zfill(d)[::-1]
        localNodeBitmap = np.array([0 if bit == '0' else 1 for bit in localNodeBinaryString])
        for dimensionToDifferentiate in range(d):
            # These are the factors x_k or (1-x_k)
            interpolationWeightsFactors = (1.0 - localNodeBitmap) + (2.0*localNodeBitmap - 1.0)*coordIndicesFpart

            # Chain rule: replace the factor that depends on the dimension we differentiate with the derivative
            interpolationWeightsFactors[..., dimensionToDifferentiate] = N[dimensionToDifferentiate]*(2.0*localNodeBitmap[dimensionToDifferentiate] - 1.0)

            # Multiply them
            interpolationWeightsForThisDimension = np.prod(interpolationWeightsFactors, axis=1)

            # Insert into interpolation weight matrix
            interpolationWeights[:, localNode, dimensionToDifferentiate] = interpolationWeightsForThisDimension

    # Compute the global indices of all the local nodes
    coordIndicesLinear = util.convertpCoordIndexToLinearIndex(N, coordIndicesIpart)
    interpolationValuesIndices = np.add.outer(coordIndicesLinear, util.elementpIndexMap(N))

    # Compute the values of the relevant nodes
    interpolationValues = cq1[interpolationValuesIndices,...]

    # Perform the interpolation sum of weight-value products
    evaluatedCQ1 = np.einsum('pil,pi...->p...l', interpolationWeights, interpolationValues)
    
    return evaluatedCQ1
