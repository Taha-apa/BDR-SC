import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
# based on : https://github.com/canyilu/Block-Diagonal-Representation-for-Subspace-Clustering written by Canyi Lu (canyilu@gmail.com)
#
#  References :
# Canyi Lu, Jiashi Feng, Tao Mei, Zhouchen Lin and Shuicheng Yan
# Subspace Clustering by Block Diagonal Representation, 
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019
# 
# tahaa.p.a78@gmail.com has rewritten this code from matlab to python using numpy

@dataclass
class modelConfig:
    lmbda : float 
    gamma : float
    knn : int
    stop_thr : float
    rho : int

def BDR_Solver(X : np.ndarray,modelConfig : modelConfig,maxIter : int = 1000) -> tuple[np.ndarray,np.ndarray]:
    lmbda,gamma,k,threshold,rho = modelConfig.__dict__.values() 
    n = X.shape[1]
    one = np.ones((n,1))
    XtX = np.matmul(X.T,X)
    I = np.identity(n)

    invXtXI = np.linalg.inv(XtX + lmbda * I)
    invXtXI[np.isnan(invXtXI)] = 0
    gammaOverLambda = gamma/lmbda

    #initializing Z,W,B
    Z = np.zeros((n,n))
    W = Z
    B = Z
    for iter in tqdm(range(0,maxIter)):
        #updating Z
        Z_old = Z
        #based on (18)
        Z = np.matmul(invXtXI,XtX + lmbda * B)
        
        #updating B
        B_old = B
        #based on propistion 7 and (19) :
        B = Z - gammaOverLambda * (np.tile(np.diag(W),(n,1)) - W)
        B = np.maximum(np.zeros(B.shape),(B + B.T) / 2)
        B = B - np.diag(np.diag(B))
        #here is the laplacian matrix
        L = np.diag(np.matmul(B,one)) - B

        #updating W
        #based on solution of (15) :
        # V : eigenvalues
        # D : eigenvectors
        D,V = np.linalg.eig(L)
        index_sorted = np.argsort(D)
        #first k indexes in the index_sorted are needed
        W =np.matmul(V[:,index_sorted[0:k]],V[:,index_sorted[0:k]].T)
        #max difference amongst values of Z and Z_old
        diff_Z = np.array(abs(Z - Z_old)).max()
        diff_B = np.array(abs(B - B_old)).max()
        diff = max([diff_Z,diff_B])
        if diff < threshold :
            break
    return B,Z

