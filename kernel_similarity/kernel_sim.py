import math
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.stats import gamma


def incomplete_cholesky(K, k, eta = 0.001):
    """
    Incomplete Cholesky decomposition. This assumes
    precalculation of the kernel matrix, wasting space,
    and some time.

    Parameters
    ------------------------
    K: the kernel matrix
    k: maximum numbers of rows for new matrix
    eta: threshold

    Return
    ------------------------
    R: new T-by-j matrix
    j: number of rows in new matrix

    """
    n = K.shape[0]
    I = []
    R = np.zeros((k,n))
    # RBF kernel diagonal is all 1
    K_diag= np.ones(n)
    ind_max=np.argmax(K_diag)
    a = K_diag[ind_max]
    I.append(ind_max)
    j = 0
    while a > eta and j < k:
        nu_j = math.sqrt(a)
        for i in range(n):
            R[j,i] = (K[I[j],i] - np.dot(R[:,i].T,R[:,I[j]]))/nu_j
        K_diag = K_diag - R[j,:]**2
        ind_max=np.argmax(K_diag)
        a = K_diag[ind_max]
        I.append(ind_max)
        j += 1
        
    return R[:j,], j


def incomplete_cholesky_kernel(X, k, sigma = None, eta = 0.001):
    """
    Incomplete Cholesky decomposition without 
    precalculation of the kernel matrix. This has less space and
    time complexity, but to estimate the median, all pairwise
    squared Euclidean distances would needed, so sigma needs to 
    be supplied.
    
    Parameters
    ------------------------
    K: the kernel matrix
    k: maximum numbers of rows for new matrix
    sigma: kernel width to use
    eta: threshold

    Return
    ------------------------
    R: new T-by-j matrix
    j: number of rows in new matrix
    """
    n = X.shape[0]
    I = []
    R = np.zeros((k,n))
    
    # RBF's maximum is 1
    K_diag= np.ones(n)
    ind_max=np.argmax(K_diag)
    a = K_diag[ind_max]
    I.append(ind_max)
    j = 0
    while j < k and a > eta:
        nu_j = math.sqrt(a)
        X_row = np.atleast_2d(X[I[j]])
        #
        K_tmp = vectorized_sq_dist(X_row,X)
        sq_dist_to_RBF(K_tmp,sigma, True)

        for i in range(n):
            R[j,i] = (K_tmp[0][i] - np.dot(R[:,i].T,R[:,I[j]]))/nu_j
        K_diag = K_diag - R[j,:]**2
        
        ind_max=np.argmax(K_diag)
        a = K_diag[ind_max]
        I.append(ind_max)
        j += 1
        
    return R[:j,], j


def vectorized_sq_dist(X,Y):
    '''
    Computes the squared euclidean on every pair of examples in
    input space. This is separated from the RBF step because
    these distances are used to determine the RBF kernel's
    parameter.
    Uses sklearn's function as it was faster than my vectorized
    version.
    
    Parameters
    ------------------------
    X: matrix 1 (n-by-m1)
    Y: matrix 2 (n-by-m2)
    
    Return
    ------------------------
    G: new n-by-n matrix of squared distances
    '''
    if len(X.shape)==1:
        X=X.reshape(-1,1)
    if len(Y.shape)==1:
        Y=Y.reshape(-1,1)
    # computes the squared distances in input space vectorized
    return pairwise_distances(X,Y,"euclidean",n_jobs=-1)**2

def sq_dist_to_RBF(G,sigma,inplace=False):
    '''
    Applies the RBF kernel to the squared distances.
    Inplace option because after this step the input
    matrix is just wasting memory space.
    
    Parameters
    ------------------------
    G: square matrix of squared euclidean distances
    sigma: kernel width
    inplace: if True, overwrites the input square matrix of distances
    
    Return
    ------------------------
    K: new n-by-n matrix of kernel distances
    '''
    
    if inplace:
        np.exp(-G/2/(sigma**2),G)
    else:
        return np.exp(-G/2/(sigma**2))
    
def estimate_kernel_width(K):
    # Set upper triangular to zero as it is a copy of lower triangular
    dists = np.tril(K)
    # Ignore zeros, input can be overwritten for speed
    return np.sqrt( 0.5 * np.median(dists[np.nonzero(dists)],overwrite_input =True))

def HSICb(X, Y):
    '''
    Calculates the biased HSIC of X and Y.
    
    Parameters
    ------------------------
    X: n-by-m1 matrix
    Y: n-by-m2 matrix
    
    Return
    ------------------------
    HSICb: the centered kernel measure of dependence
    '''
    n=X.shape[0]
    # Calculate squared distance matrix
    K = vectorized_sq_dist(X,X)
    # Estimate kernel width
    width_X = estimate_kernel_width(K)
    # Reuse distance matrices from kernel width estimation
    # Apply RBF in place
    sq_dist_to_RBF(K,width_X, True)
    # Center one of the kernels
    H = np.identity(n) - np.ones((n,n), dtype = float) / n
    K=np.dot(np.dot(H, K), H)
    
    # Repeat for other variable
    L = vectorized_sq_dist(Y,Y)
    width_Y = estimate_kernel_width(L)
    sq_dist_to_RBF(L,width_Y, True)
    
    # This is the biased estimate!
    # Efficient calculation of trace of dot product
    return np.einsum('ij,ji->', K, L)/n**2
    
def HSICb_approx(X, Y, k=100, sigmas=None, precalc=True):
    '''
    Calculates the biased HSIC of X and Y using incomplete
    Cholseky approximation.
    
    Parameters
    ------------------------
    X: n-by-m1 matrix
    Y: n-by-m2 matrix
    k: number of factors to retain
    sigma: kernel width to use when kernel
           is not precalculated
    precalc: if False kernel matrices are calculated
             row-by-row when needed to save space
    
    Return
    ------------------------
    HSICb: the centered kernel measure of dependence
    '''
    n=X.shape[0]
    if precalc:
        # Calculate squared distance matrix
        K = vectorized_sq_dist(X,X)
        # Estimate kernel width
        width_X = estimate_kernel_width(K)
        # Reuse distance matrices from kernel width estimation
        # Apply RBF in place
        sq_dist_to_RBF(K,width_X, True)
        # Take approximation of K
        K,_=incomplete_cholesky(K,k)

        # Repeat for other variable
        L = vectorized_sq_dist(Y,Y)
        width_Y = estimate_kernel_width(L)
        sq_dist_to_RBF(L,width_Y, True)
        L,_=incomplete_cholesky(L,k)
    else:
        assert(sigmas is not None)
        K,_=incomplete_cholesky_kernel(X,k,sigmas[0])
        L,_=incomplete_cholesky_kernel(Y,k,sigmas[1])
    
    # Center one of the lower triangles
    K_c = K.T - K.T.mean(axis=0)
    # Intermediate result
    inter = L * np.mat(K_c)
    # This is the biased estimate!
    return np.trace(inter * inter.T)/n**2


def HSICb_test(X, Y, signif = 0.05):
    '''
    Calculates the biased test statistic, and threshold given a
    significance level.
    
    Parameters
    ------------------------
    X: n-by-m1 matrix
    Y: n-by-m2 matrix
    signif: the significance level
    
    Return
    ------------------------
    test_statistic: the biased HSIC test statistic
    thresh: the threshold for the given significance level
            based on th gamma distrobution
    '''
    n=X.shape[0]
    # calculate squared distance matrices
    G = vectorized_sq_dist(X,X)
    Q = vectorized_sq_dist(Y,Y)
    # Estimate kernel widths
    width_X = estimate_kernel_width(G)
    width_Y = estimate_kernel_width(Q)
    # Apply RBF
    K=sq_dist_to_RBF(G,width_X)
    L=sq_dist_to_RBF(Q,width_Y)
    H = np.identity(n) - np.ones((n,n), dtype = float) / n
    # center BOTH kernels
    K_c=np.dot(np.dot(H, K), H)
    L_c=np.dot(np.dot(H, L), H)
    # note that test statistics is n*HSICb!
    # efficient calculation of trace of dot product, sum is inplace
    # L_c is not needed here, could be L too
    test_stat= np.einsum('ij,ji->', K_c, L_c)/n
    
    var_HSIC = (K_c * L_c / 6)**2
    var_HSIC = (np.sum(var_HSIC) - np.trace(var_HSIC) ) / n / (n-1)
    var_HSIC = var_HSIC * 72 * (n-4) * (n-5) / n / (n-1) / (n-2) / (n-3)

    K = K - np.diag(np.diag(K))
    L = L - np.diag(np.diag(L))
    
    ones = np.ones((n, 1), dtype = float)
    mu_X = np.dot(np.dot(ones.T, K), ones) / n / (n-1)
    mu_Y = np.dot(np.dot(ones.T, L), ones) / n / (n-1)

    mu_HSIC = (1 + mu_X * mu_Y - mu_X - mu_Y) / n

    alpha = mu_HSIC**2 / var_HSIC
    beta = var_HSIC*n / mu_HSIC

    thresh = gamma.ppf(1-signif, alpha, scale=beta)[0][0]

    return (test_stat, thresh)

def CKA(X, Y, approx=False, k=None, eta=None):
    '''
    Calculates the CKA of X and Y.
    
    Parameters
    ------------------------
    X: n-by-m1 matrix
    Y: n-by-m2 matrix
    approx: if approx=True it performs incomplete
            Cholesky decomposition
    
    Return
    ------------------------
    CKA: the standardized kernel measure of dependence
    '''
    n=X.shape[0]
    # Calculate squared distance matrix
    K = vectorized_sq_dist(X,X)
    # Estimate kernel width
    width_X = estimate_kernel_width(K)
    # Reuse distance matrices from kernel width estimation
    # Apply RBF in place
    sq_dist_to_RBF(K,width_X, True)
    
    # Repeat for other variable
    L = vectorized_sq_dist(Y,Y)
    width_Y = estimate_kernel_width(L)
    sq_dist_to_RBF(L,width_Y, True)
    
    H = np.identity(n) - np.ones((n,n), dtype = float) / n
    if approx:
        assert(k is not None)
        assert(eta is not None)
        K,_=incomplete_cholesky(K,k,eta)
        L,_=incomplete_cholesky(L,k,eta)
        # Center one of the lower triangles
        K_c = K.T - K.T.mean(axis=0)
        inter = L * np.mat(K_c)
        HSIC_XY = np.trace(inter * inter.T)/n**2
        inter = K * np.mat(K_c)
        HSIC_XX = np.trace(inter * inter.T)/n**2
        L_c = L.T - L.T.mean(axis=0)
        inter = L * np.mat(L_c)
        HSIC_YY = np.trace(inter * inter.T)/n**2
    else:
        # Center one of the kernels
        H = np.identity(n) - np.ones((n,n), dtype = float) / n
        K=np.dot(np.dot(H, K), H)
        # Biased HSIC estimates
        # Efficient calculation of trace of dot product
        HSIC_XY=np.einsum('ij,ji->', K, L)/n**2
        # Reuse K
        HSIC_XX=np.einsum('ij,ji->', K, K)/n**2
        # Center L
        L=np.dot(np.dot(H, L), H)
        HSIC_YY=np.einsum('ij,ji->', L, L)/n**2
        
    return  HSIC_XY/((HSIC_XX**0.5)*(HSIC_YY**0.5))

def select_feature_CKA(array,k,approx=False, k_approx=None, eta=0.001):
    '''
    Selects best k sensors based on CKA in a
    greedy fashion.
    ------------------
    inputs:
    array: m x n array of sensor measurements
    k: number of sensors to select
    ------------------
    return: list of column indices of to select
            sorted by selection order
    '''
    m=array.shape[1]
    unselected=list(range(m))
    selected=[]
    sim_sequence=[]
    # find first sensor
    sim_vector=np.zeros(m)
    for i in unselected:
        # calculate sensor's CKA with rest of sensors
        sim_vector[i]=CKA(array[:,i],np.delete(array, i, axis=1),approx,k_approx, eta)
    # select most similar to unselected
    best=np.argmax(sim_vector)
    unselected.remove(best)
    selected.append(best)
    sim_sequence.append(sim_vector[best])
    print("feature 0 found")
    # find next k sensors
    while len(selected)<k:
        sim_vector=np.full(m,2.0)
        # go over leftovers
        for i in unselected:
            sim_vector[i]=CKA(array[:,i],array[:,selected],approx,k_approx, eta)
        # select least similar to selected
        best=np.argmin(sim_vector)
        unselected.remove(best)
        selected.append(best)
        sim_sequence.append(sim_vector[best])
        print("feature %d found"%(len(selected)-1))
    return selected, sim_sequence