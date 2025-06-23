#--------------------------------------------------------------------------
# This function takes a NxN coefficient matrix and returns a NxN adjacency
# matrix by choosing the K strongest connections in the similarity graph
# CMat: NxN coefficient matrix
# K: number of strongest edges to keep; if K=0 use all the exiting edges
# CKSym: NxN symmetric adjacency matrix
#--------------------------------------------------------------------------
# Copyright @ Ehsan Elhamifar, 2012
#--------------------------------------------------------------------------
# tahaa.p.a78@gmai.com has rewritten this code from matlab to python using numpy library

import numpy as np
def BuildAdjacency(coefficient_matrix : np.ndarray,k: int = 0) -> tuple[np.ndarray,np.ndarray]:
    n = len(coefficient_matrix)
    eps = np.finfo(float).eps
    coefficient_mat_abs = abs(coefficient_matrix)
    #sorted in descending order
    index_CMAT = np.argsort(coefficient_mat_abs,axis=0)[::-1]
    if k == 0:
        for i in range(0,n):
            coefficient_mat_abs[:,i] = coefficient_mat_abs[:,i] / (coefficient_mat_abs[index_CMAT[1,i],i] + eps)
    else:
        for i in range(0,n):
            for j in range(0,k):
                coefficient_mat_abs[index_CMAT[j,i],i] = coefficient_mat_abs[index_CMAT[j,i],i] / (coefficient_mat_abs[index_CMAT[1,i],i] + eps)
    Adjacency_Sym_matrix = coefficient_mat_abs + coefficient_mat_abs.T
    return Adjacency_Sym_matrix,coefficient_mat_abs


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
#--------------------------------------------------------------------------
# Copyright @ Ehsan Elhamifar, 2012
#--------------------------------------------------------------------------
# tahaa.p.a78@gmail.com has rewritten this code from matlab to python using numpy
import numpy as np
def thrC(C : np.ndarray, ro : int = 1) :
    if ro < 1:
    
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S,Ind = np.sort(abs(C),axis=0)[-1:,:],C.argsort(axis=0)[::-1]
        for i in range(0,N):
            cL1 = sum(S[:,i])
            stop = False
            cSum = 0
            t = -1
            while not stop:
                t +=1
                cSum += S[t,i]
                if cSum >= ro * cL1 :
                    stop = True 
                    Cp[Ind[0:t,i],i] = C[Ind[0:t,i],i]
    else :
        Cp = C
    return Cp