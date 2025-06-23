#--------------------------------------------------------------------------
# This function takes the groups resulted from spectral clutsering and the
# ground truth to compute the misclassification rate.
# groups: [grp1,grp2,grp3] for three different forms of Spectral Clustering
# s: ground truth vector
# Missrate: 3x1 vector with misclassification rates of three forms of
# spectral clustering
#--------------------------------------------------------------------------
# Copyright @ Ehsan Elhamifar, 2012
#--------------------------------------------------------------------------
# tahaa.p.a78@gmai.com has rewritten this code from matlab to python using numpy and itertools libraries
import numpy as np
from itertools import permutations as perms
def missRate(groups : np.ndarray,s : np.ndarray) -> np.ndarray:
    num_spect = len(groups)
    missrate = np.array((num_spect,1))
    n = max(s)
    for row in range(groups.shape[1]):
        missrate[row,1] = missClassGroups(groups[:,row],s,n) / len(s)
    return missrate

def perm_to_np(perm : perms) -> np.ndarray:
    perms_np = []
    for p in perm :
        perms_np.append(p)
    return perms_np

#-------------------------------------------------------------------------
# [miss,index] = missclass(Segmentation,RefSegmentation,ngroups)
# Computes the number of missclassified points in the vector Segmentation. 
# Segmentation: 1 by sum(npoints) or sum(ngroups) by 1 vector containing 
# the label for each group, ranging from 1 to n
# npoints: 1 by ngroups or ngroups by 1 vector containing the number of 
# points in each group.
# ngroups: number of groups
#--------------------------------------------------------------------------
# Copyright @ Ehsan Elhamifar, 2012
#--------------------------------------------------------------------------
# tahaa.p.a78@gmai.com has rewritten this code from matlab to python using numpy library

def missClassGroups(seg : np.ndarray,RefSeg : np.ndarray,nGroups : int) -> tuple[np.ndarray,np.ndarray] :
    permutations = perm_to_np(perms(range(0,nGroups)))
    if(seg.shape[1] == 1):
        seg = seg.T
    miss = np.zeros((permutations.shape[0],seg.shape[0]))
    for i in range(len(seg)):
        for j in range(len(permutations)):
            miss[i,j] = sum(np.array(seg[i,:] != permutations[i,RefSeg]))
    missAndTemp = np.min(miss,0)
    index = permutations[missAndTemp[1],:]
    miss = missAndTemp[0]
    return miss,index