import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg

import sys
import time
import util

def linSolve(K, c):
    if np.size(K,0) > 2e5:
        linSolver = 'cg'
    else:
        linSolver = 'spsolve'

    if linSolver == 'spsolve':
        print 'Using spsolve'
        x = sparse.linalg.spsolve(K, c)
    elif linSolver == 'cg':
        print 'Using CG'
        DHalfInvDiag = 1./np.sqrt(K.diagonal())
        DHalfInv = sparse.diags([DHalfInvDiag], offsets=[0])
        B = DHalfInv*(K*DHalfInv)
        d = DHalfInv*c

        def cgCallback(xk):
            print np.linalg.norm(B*xk-d)
        
        y, info = sparse.linalg.minres(B, d, callback=cgCallback)
        print 'info = {}'.format(info)
        x = DHalfInv*y
    elif linSolver == 'cholesky':
        print 'Using cholesky'
        cholK = cholesky(K)
        x = cholK.solve_A(c)
    return x

def saddleDirect(A, B, rhsList, fixed):
    K = sparse.bmat([[A, B.T],
                     [B, None]], format='csc')

    xList = []
    for rhs in rhsList:
        b = np.zeros(K.shape[0])
        b[:np.size(rhs)] = rhs
        xAll = sparse.linalg.spsolve(K, b, use_umfpack=False)
        xList.append(xAll[:np.size(rhs)])

    return xList


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

def saddleNullSpaceGeneralBasis(A, B, S, rhsList, coarseNodes):
    Np = A.shape[0]
    Nc = np.size(coarseNodes)

    coarseNodesMask = np.zeros(Np, dtype='bool')
    coarseNodesMask[coarseNodes] = True
    notCoarseNodesMask = np.logical_not(coarseNodesMask)
    notCoarseNodes = np.where(notCoarseNodesMask)[0]
    nodePermutation = np.hstack([coarseNodes, notCoarseNodes])

    Sc   = S[:,coarseNodesMask]
    PSub = Sc[notCoarseNodesMask,:]
    Snc  = S[:,notCoarseNodesMask]
    SPerm = sparse.bmat([[Sc, Snc]], format='csc')

    I = sparse.identity(Np-Nc, format='csc')
    Bn = B*Snc

    Z = sparse.bmat([[-Bn],
                     [I]], format='csc')

    ZT = Z.T
    SPermT = SPerm.T
   
    class mutableClosure:
        Atimer = 0
        Mtimer = 0
        counter = 0
        
    def Ax(x):
        start = time.time()
        y = ZT*(SPermT*(A*(SPerm*(Z*x))))
        end = time.time()
        mutableClosure.Atimer += end-start
        mutableClosure.counter += 1
        return  y

    # Does not work...
    def MInvx(x):
        start = time.time()
        y = PSub*(PSub.T*x) + x
        end = time.time()
        mutableClosure.Mtimer += end-start
        return  y

    ALinearOperator = sparse.linalg.LinearOperator(dtype='float64', shape=(Np-Nc, Np-Nc), matvec=Ax)
    MInvLinearOperator = sparse.linalg.LinearOperator(dtype='float64', shape=(Np-Nc, Np-Nc), matvec=MInvx)
    
    correctorList = []
    for rhs in rhsList:
        b = Z.T*(SPerm.T*rhs)

        def cgCallback(xk):
            print np.linalg.norm(Z.T*(SPerm.T*(A*(SPerm*(Z*xk))))-b)
            return
        
        mutableClosure.counter = 0
        mutableClosure.Atimer = 0
        mutableClosure.Mtimer = 0
        start = time.time()
        xPerm,info = sparse.linalg.cg(ALinearOperator, b, callback=None, tol=1e-9, M=None)
        #print mutableClosure.counter, mutableClosure.Atimer, mutableClosure.Mtimer
        end = time.time()
        #print end-start
        
        if info != 0:
            raise(FailedToConverge('CG failed to converge, info={}'.format(info)))

        totalDofs = A.shape[0]
        corrector = np.zeros(Np)
        corrector = SPerm*(Z*xPerm)
        correctorList.append(corrector)
        
    return correctorList

def saddleNullSpaceHierarchicalBasis(A, B, P, rhsList, coarseNodes, fixed):
    '''Solve ( S'*A*S  S'*B' ) ( y  )   ( S'b )
             (    B*S  0     ) ( mu ) = ( 0   )

    and compute x = S*y where

        ( |  0 )
    S = ( P    ) 
        ( |  I )

    if the nodes are reordered so that coarseNodes comes first.
    '''
    Np = A.shape[0]
    Nc = np.size(coarseNodes)

    if fixed is not None:
        raise(NotImplementedError('Boundary conditions not implemented here yet....'))
    
    coarseNodesMask = np.zeros(Np, dtype='bool')
    coarseNodesMask[coarseNodes] = True
    notCoarseNodesMask = np.logical_not(coarseNodesMask)
    notCoarseNodes = np.where(notCoarseNodesMask)[0]
    nodePermutation = np.hstack([coarseNodes, notCoarseNodes])
    
    PSub = P[notCoarseNodes,:]
    I1 = sparse.identity(Nc, format='csc')
    I2 = sparse.identity(Np-Nc, format='csc')
    Bn = B[:,notCoarseNodesMask]

    S = sparse.bmat([[I1,   None],
                     [PSub, I2]], format='csc')
    Z = sparse.bmat([[-Bn],
                     [I2]], format='csc')

    ST = S.T
    ZT = Z.T
    PSubT = PSub.T
    APerm = A[nodePermutation][:,nodePermutation]
    
    class mutableClosure:
        Atimer = 0
        Mtimer = 0
        counter = 0
        
    def Ax(x):
        start = time.time()
        y = ZT*(ST*(APerm*(S*(Z*x))))
        end = time.time()
        mutableClosure.Atimer += end-start
        mutableClosure.counter += 1
        return  y

    def MInvx(x):
        start = time.time()
        y = PSub*(PSubT*x) + x
        end = time.time()
        mutableClosure.Mtimer += end-start
        return  y
    
    ALinearOperator = sparse.linalg.LinearOperator(dtype='float64', shape=(Np-Nc, Np-Nc), matvec=Ax)
    MInvLinearOperator = sparse.linalg.LinearOperator(dtype='float64', shape=(Np-Nc, Np-Nc), matvec=MInvx)
    
    correctorList = []
    for rhs in rhsList:
        #print '.',
        b = ZT*(ST*rhs[nodePermutation])

        def cgCallback(xk):
            print str(np.size(xk)) + '  ' + str(np.linalg.norm(ZT*(ST*(APerm*(S*(Z*xk))))-b))
            return
        
        mutableClosure.counter = 0
        mutableClosure.Atimer = 0
        mutableClosure.Mtimer = 0
        start = time.time()
        xPerm,info = sparse.linalg.cg(ALinearOperator, b, callback=cgCallback, tol=1e-9, M=MInvLinearOperator)
        #print mutableClosure.counter, mutableClosure.Atimer, mutableClosure.Mtimer
        end = time.time()
        #print end-start
        if info != 0:
            raise(FailedToConverge('CG failed to converge, info={}'.format(info)))

        totalDofs = A.shape[0]
        corrector = np.zeros(Np)
        corrector[nodePermutation] = S*(Z*xPerm)
        correctorList.append(corrector)
        
    return correctorList
    
    # We have that (B*S)[coarseNodes,coarseNodes] is identity.  A
    # "nice basis" is found for all projections.
    
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

    class mutableClosure:
        timer = 0
        counter = 0
        
    def Ax(x):
        start = time.time()
        y = A21*(Btilde*x) + A22*x + Btildecsc.T*(A11*(Btilde*x)) + Btildecsc.T*(A12*x)
        end = time.time()
        mutableClosure.timer += end-start
        mutableClosure.counter += 1
        return  y

    ALinearOperator = sparse.linalg.LinearOperator(dtype='float64', shape=A22.shape, matvec=Ax)
    
    correctorList = []
    for rhs in rhsList:
        #print '.',
        b = rhs[notCoarseNodesMask] + Btildecsc.T*rhs[coarseNodesMask]

        mutableClosure.counter = 0
        mutableClosure.timer = 0
        x,info = sparse.linalg.cg(ALinearOperator, b, tol=1e-9)
        print mutableClosure.counter, mutableClosure.timer
        
        if info != 0:
            raise(FailedToConverge('CG failed to converge, info={}'.format(info)))

        totalDofs = A.shape[0]
        corrector = np.zeros(totalDofs)
        corrector[notCoarseNodesMask] = x
        corrector[coarseNodesMask] = Btilde*x
        correctorList.append(corrector)
        
    return correctorList


from scikits.sparse.cholmod import cholesky
from scikits.sparse.cholmod import analyze
            
def solveWithBlockDiagonalPreconditioner(A, B, bList):
    """Solve saddle point problem with block diagonal preconditioner

    / A  B.T \   / r \   / b \
    |        | * |   | = |   |
    \ B   0  /   \ s /   \ 0 /

    Section 10.1.1 in "Numerical solution of saddle point problems",
    Benzi, Golub and Liesen.
    """

    n = np.size(A,0)
    m = np.size(B,0)

    cholA = cholesky(A)
    S = np.zeros((B.shape[0], B.shape[0]))
    for s, Brow in zip(S.T, B):
        y = cholA.solve_A(np.array(Brow.todense()).T.squeeze())
        s[:] = -B*y
        
    SInv = np.linalg.inv(S)

    def solveP(x):
        r = x[:n]
        s = x[-m:]
        rSol = cholA.solve_A(r)
        sSol = -np.dot(SInv, s)
        return np.hstack([rSol, sSol])

    M = sparse.linalg.LinearOperator((n+m,n+m), solveP)

    K = sparse.bmat([[A, B.T],
                     [B, None]], format='csc');
    c = np.zeros(n+m)

    numIter = [0]
    def cgCallback(x):
        numIter[0] +=  1

    rList = []
    xList = []
    infoList = []
    numIterList = []
    for b in bList:
        numIter = [0]
        c[:n] = b
        x, info = sparse.linalg.cg(K, c, tol=1e-9, M=M, callback=cgCallback)
        r = x[:n]
        rList.append(r)
        xList.append(x)
        infoList.append(info)
        numIterList.append(numIter[0])

    return rList

class choleskyCache:
    def __init__(self, NMax):
        self.NMax = NMax
        self.indexBasis = util.linearpIndexBasis(NMax)
        self.factorCache = dict()

    def lookup(self, N, A):
        print 'lookup'
        index = np.dot(self.indexBasis, N)
        if index not in self.factorCache:
            print 'miss'
            t = time.time()
            self.factorCache[index] = analyze(A)
            t = time.time()-t
            print 'a', t
        else:
            print 'hit'
        cholAFactor = self.factorCache[index]
        t = time.time()
        cholAFactorReturn = cholAFactor.cholesky(A)
        t = time.time()-t
        print 'c', t
        return cholAFactorReturn

def schurComplementSolve(A, B, bList, fixed, NPatchCoarse=None, NCoarseElement=None, cholCache=None):
    correctorFreeList = []

    # Pre-processing (page 12, Engwer, Henning, Malqvist)
    # Rename for brevity
    A = imposeBoundaryConditionsStronglyOnMatrix(A, fixed)
    bList = [imposeBoundaryConditionsStronglyOnVector(b, fixed) for b in bList]
    B = imposeBoundaryConditionsStronglyOnInterpolation(B, fixed)

    # Compute Y
    #luA = sparse.linalg.splu(A)
    if False and cholCache is not None:
        assert(NPatchCoarse is not None)
        assert(NCoarseElement is not None)
        N = NPatchCoarse*NCoarseElement
        print A.shape
        print A.nnz
        cholA = cholCache.lookup(N, A)
    else:
        cholA = cholesky(A)

    Y = np.zeros((B.shape[1], B.shape[0]))
    for y, c in zip(Y.T, B):
        #y[:] = luA.solve(np.array(c.todense()).T.squeeze())
        #y[:] = luA_approx.solve(np.array(c.todense()).T.squeeze())
        y[:] = cholA.solve_A(c.T).todense().squeeze()

    S = B*Y
    invS = np.linalg.inv(S)

    # Post-processing
    for b in bList:
        r = b
        #q = luA.solve(r)
        #q = luA_approx.solve(r)
        q = cholA.solve_A(r)
        lam = np.dot(invS, B*q)
        correctorFree = q - np.dot(Y,lam)
        correctorFreeList.append(correctorFree)
    return correctorFreeList


# Remove fized degrees of freedom from matrices A, B and b
def imposeBoundaryConditionsStronglyOnMatrix(A, fixed):
    AStrong = A.copy()
    nzFixedCols = AStrong[:,fixed].nonzero()
    AStrong[nzFixedCols[0],fixed[nzFixedCols[1]]] = 0
    nzFixedRows = AStrong[fixed,:].nonzero()
    AStrong[fixed[nzFixedRows[0]],nzFixedRows[1]] = 0
    AStrong[fixed,fixed] = 1
    return AStrong

def imposeBoundaryConditionsStronglyOnVector(b, fixed):
    bStrong = b.copy()
    bStrong[fixed] = 0
    return bStrong

def imposeBoundaryConditionsStronglyOnInterpolation(B, fixed):
    BStrong = B.copy()
    nzFixedCols = BStrong[:,fixed].nonzero()
    BStrong[nzFixedCols[0],fixed[nzFixedCols[1]]] = 0
    BStrong.eliminate_zeros()
    BStrong = BStrong[BStrong.getnnz(1)>0]
    return BStrong

