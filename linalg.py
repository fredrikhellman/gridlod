import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

import sys
import time

def saddle(A, B, rhsList):
    ''' saddle

    Solve (A B'; B 0) = rhs
    '''
    xList = []

    # Pre-processing (page 12, Engwer, Henning, Malqvist)
    # Rename for brevity
    from scikits.sparse.cholmod import cholesky
            
    # Compute Y
    print "A"
    sys.stdout.flush()
    
    cholA = cholesky(A)
    Y = np.zeros((B.shape[1], B.shape[0]))
    for y, b in zip(Y.T, B):
        print ".",
        sys.stdout.flush()
        
        y[:] = cholA.solve_A(np.array(b.todense()).T.squeeze())

    print "B"
    sys.stdout.flush()
        
    S = B*Y
    invS = np.linalg.inv(S)

    print "C"
    sys.stdout.flush()

    # Post-processing
    for rhs in rhsList:
        q   = cholA.solve_A(rhs)
        lam = np.dot(invS, B*q)
        x   = q - np.dot(Y,lam)
        xList.append(x)

    print "D"
    sys.stdout.flush()
        
    return xList

class FailedToConverge(Exception):
    pass

def saddleNullSpace(A, B, rhsList, coarseNodes):
    '''lodSaddle

    Solve (A B'; B 0) = rhs

    Use a null space method. We assume the columns of B can be
    permuted with P so that
    
    BP = [D Bn]

    where D is diagonal. This is the case when the interpolation
    operator includes no other coarse node than its own in its nodal
    variable definition.

    It is also possible (see e.g. M. Benzi, G. H. Golub and J. Liesen)
    to make a permutation

    BP = [Bb Bn]

    where Bb is invertible. However, this requires to compute
    Bb^(-1)*Bn...

    '''
    
    ## Find cols with one non-zero only
    Bcsr = B.tocsr()

    # Eliminate zero or almost-zero rows
    Bcsr.data[np.abs(Bcsr.data)<1e-12] = 0
    Bcsr.eliminate_zeros()
    Bcsr  = Bcsr[np.diff(Bcsr.indptr) != 0,:]

    Bcsc = Bcsr.tocsc()
    
    coarseNodesMask = np.zeros(Bcsc.shape[1], dtype='bool')
    coarseNodesMask[coarseNodes] = True
    notCoarseNodesMask = np.logical_not(coarseNodesMask)
    Bb = Bcsc[:,coarseNodesMask]
    Bn = Bcsc[:,notCoarseNodesMask]

    def diagonalCsc(A):
        n = A.shape[0]
        if A.shape[1] != n:
            return None
        if np.all(A.indptr != np.arange(n+1)):
            return None
        if np.all(A.indices != np.arange(n)):
            return None
        else:
            return sparse.dia_matrix((A.data, 0), shape=(n,n))
    
    Bbdiag = diagonalCsc(Bb)
    if Bbdiag is None:
        raise(NotImplementedError('Can''t handle general interpolation ' +
                                  'operators. Needs to be easy to find its null space...'))
        
    BbInv = Bbdiag.copy()
    BbInv.data = 1./BbInv.data

    Btildecsc = -BbInv*Bn
    Btildecsc.sort_indices()
    
    # For faster MV-multiplication
    Btilde = Btildecsc.tocsr()
    Btilde.sort_indices()
    
    A11 = A[coarseNodesMask][:,coarseNodesMask].tocsr()
    A11.sort_indices()
    A12 = A[coarseNodesMask][:,notCoarseNodesMask].tocsr()
    A12.sort_indices()
    A22 = A[notCoarseNodesMask][:,notCoarseNodesMask].tocsr()
    A22.sort_indices()
    A21 = A12.T.tocsr()
    A21.sort_indices()

    class mutable_closure:
        timer = 0
        counter = 0
        
    def Ax(x):
        start = time.time()
        y = A21*(Btilde*x) + A22*x + Btildecsc.T*(A11*(Btilde*x)) + Btildecsc.T*(A12*x)
        end = time.time()
        mutable_closure.timer += end-start
        mutable_closure.counter += 1
        return  y

    ALinearOperator = sparse.linalg.LinearOperator(dtype='float64', shape=A22.shape, matvec=Ax)
    
    correctorList = []
    for rhs in rhsList:
        print '.',
        b = rhs[notCoarseNodesMask] + Btildecsc.T*rhs[coarseNodesMask]
        x,info = sparse.linalg.cg(ALinearOperator, b, tol=1e-9)
        print mutable_closure.counter, mutable_closure.timer
        if info != 0:
            raise(FailedToConverge('CG failed to converge, info={}'.format(info)))

        totalDofs = A.shape[0]
        corrector = np.zeros(totalDofs)
        corrector[notCoarseNodesMask] = x
        corrector[coarseNodesMask] = Btilde*x
        correctorList.append(corrector)
        
    return correctorList
