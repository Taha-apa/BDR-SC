from BDR_Solver import BDR_Solver,modelConfig
from measure import Missclassification
from utils import BuildAdjacency,DataProjection,thrC
from sklearn.cluster import SpectralClustering
import numpy as np
import sys

sys.path.append('model')

def save_affinities(CKsym_B : np.ndarray,
                    CKsym_Z : np.ndarray,
                    out_dir : str,
                    modelparam_name : str,
                    dataset_name :str,
                    n_clusters :int
                    ):
    np.save(f'{out_dir}/aff_{dataset_name}_{modelparam_name}_k{n_clusters}_B.npy',CKsym_B)
    np.save(f'{out_dir}/aff_{dataset_name}_{modelparam_name}_k{n_clusters}_Z.npy',CKsym_Z)
    
#TO BE DONE
def validate(grps_pred,grps_grnd):
    pass
def run(
    modelparam: modelConfig,
    modelparam_name : str,
    dataset_name : str,
    maxIter: int,
    n_clusters: int,
    data: np.ndarray,
    visualize : bool = False,
    save_affnt : bool = True,
    out_dir : str = 'results',
):
    X_adj = DataProjection(data)
    B,Z = BDR_Solver(X_adj,k = modelparam.knn,lmbda=modelparam.lmbda,gamma=modelparam.gamma,threshold=modelparam.stop_thr,maxIter=maxIter)
    CKsym_B,_ = BuildAdjacency(thrC(B,modelparam.rho))
    grps_B = SpectralClustering(n_clusters=n_clusters,affinity='precomputed').fit_predict(CKsym_B)
    
    CKsym_Z,_ = BuildAdjacency(thrC(Z,modelparam.rho))
    grps_Z = SpectralClustering(n_clusters=n_clusters,affinity='precomputed').fit_predict(CKsym_Z)

    #TO BE DONE
    if visualize:
        pass
    if save_affnt:save_affinities(CKsym_B,CKsym_Z,out_dir,modelparam_name,dataset_name,n_clusters)
    return grps_B,grps_Z


