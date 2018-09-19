import util
import numpy as np
    
def evaluateP0(N, p0, coordinates):
    """Compute a piecewise constant (p0) function on an N-grid at the
       given coordinates. The function can be tensor-valued.
    """

    assert(np.all(coordinates <= 1.0) and np.all(coordinates >= 0.0))
    assert(p0.shape[0] == np.prod(N))
    
    eps = np.finfo(np.float64).eps

    # Find what index coordinates the coordinates are in
    coordIndicesDecimal = coordinates.astype(np.float64)*N
    coordIndicesDecimal *= (1.-eps) # This guarantees we can call with coordinate 1.0
    coordIndices = np.floor(coordIndicesDecimal).astype(np.int64)

    # Convert to linear indices
    linearIndices = util.convertpCoordIndexToLinearIndex(N-1, coordIndices)

    # Lookup function values
    evaluatedP0 = p0[linearIndices,...]
    
    return evaluatedP0

def evaluateP1(N, p1, coordinates):
    """Compute a piecewise constant (p1) function on an N-grid at the
       given coordinates. The function can be tensor-valued.

    d-linear interpolation is used, i.e. linear in 1d, bilinear in 2d
    etc.

    """

    assert(np.all(coordinates <= 1.0) and np.all(coordinates >= 0.0))
    assert(p1.shape[0] == np.prod(N+1))
    
    eps = np.finfo(np.float64).eps

    d = N.size
    
    # Find what index coordinates the coordinates are in
    coordIndicesDecimal = coordinates.astype(np.float64)*N
    coordIndicesDecimal *= (1.-eps) # This guarantees we can call with coordinate 1.0
    
    # The integer part tells in what element to perform the interpolation
    # The fractional part gives the interpolation weights within the element
    coordIndicesIpart = np.floor(coordIndicesDecimal).astype(np.int64)
    coordIndicesFpart = coordIndicesDecimal - coordIndicesIpart

    # Compute interpolation weights
    interpolationWeights = np.zeros((coordinates.shape[0], 2**d))
    for localNode in range(2**d):
        localNodeBinaryString = bin(localNode)[2:].zfill(d)[::-1]
        localNodeBitmap = np.array([0 if bit is '0' else 1 for bit in localNodeBinaryString])

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
    interpolationValues = p1[interpolationValuesIndices,...]

    # Perform the interpolation sum of weight-value products
    evaluatedP1 = np.einsum('pi...,pi...->p...', interpolationWeights, interpolationValues)
    
    return evaluatedP1
