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


    