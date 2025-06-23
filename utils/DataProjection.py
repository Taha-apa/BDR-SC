#--------------------------------------------------------------------------
# This function takes the D x N data matrix with columns indicating
# different data points and project the D dimensional data into a r
# dimensional subspace using PCA.
# X: D x N matrix of N data points
# r: dimension of the PCA projection, if r = 0, then no projection
# Xp: r x N matrix of N projectred data points
#--------------------------------------------------------------------------
# Copyright @ Ehsan Elhamifar, 2012
#--------------------------------------------------------------------------
# tahaa.p.a78@gmai.com has rewritten this code from matlab to python using numpy library

import numpy as np

def DataProjection(X : np.ndarray,r : int = 0 ) -> np.ndarray :
    X_projected = []
    if r == 0 : 
        X_projected = X
    else :
        U,_,_ = np.linalg.svd(X)
        X_projected = np.matmul(U[:,1:r].T,X)
    return np.array(X_projected)