import numpy as np
#based on : https://github.com/canyilu/Block-Diagonal-Representation-for-Subspace-Clustering
#References :
#Canyi Lu, Jiashi Feng, Tao Mei, Zhouchen Lin and Shuicheng Yan
# Subspace Clustering by Block Diagonal Representation, 
# IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019

def BDR_Solver(X : np.ndarray,k : int,lmbda : float,gamma : float,threshold : float = 1e-3,maxIter : int = 1000):
    n = len(X)
    one = np.ones((n,1))
    XtX = np.matmul(X.T,X)
    I = np.identity(n)
    invXtXI = I / (XtX + lmbda * I)
    gammaOverLambda = gamma/lmbda

    #initializing Z,W,B
    Z = np.zeros(n)
    W = Z
    B = Z
    for iter in range (0,maxIter):
        #updating Z
        Z_old = Z
        #based on (18)
        Z = np.matmul(invXtXI,XtX + lmbda * B)
        
        #updating B
        B_old = B
        #based on propistion 7 and (19) :
        B = Z - gammaOverLambda * (np.tile(np.diag(W),(1,n)) - W)
        B = max(0,(B + B.T) / 2)
        B = B - np.diag(np.diag(B))
        #here is the laplacian matrix
        L = np.diag(np.matmul(B,one)) - B

        #updating W
        #based on solution of (15) :
        # V : eigenvalues
        # D : eigenvectors
        V,D = np.linalg.eig(L)
        #changing D to diagoal matrix
        D = np.diag(D)   
        index_sorted = np.argsort(D)
        #first k indexes in the index_sorted are needed
        W =np.matmul(V[:,index_sorted[0:k]],V[:,index_sorted[0:k]].T)
        #max difference amongst values of Z and Z_old
        diff_Z = max(max(abs(Z - Z_old)))
        diff_B = max(max(abs(B - B_old)))
        diff = max([diff_Z,diff_B])
        if diff < threshold :
            break
        
