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