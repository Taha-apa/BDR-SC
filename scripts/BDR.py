from model.BDR_Solver import BDR_Solver,modelConfig
from measure import Missclassification
from utils import BuildAdjacency,DataProjection,thrC
from sklearn.cluster import SpectralClustering
import numpy as np
import sys

sys.path.append('model')

def save_affinities(CKsym_B : np.ndarray,
                    CKsym_Z : np.ndarray,
                    aff_dir : str,
                    modelparam_name : str,
                    dataset_name :str,
                    n_clusters :int
                    ):
    np.save(f'{aff_dir}/aff_{dataset_name}_{modelparam_name}_k{n_clusters}_B.npy',CKsym_B)
    np.save(f'{aff_dir}/aff_{dataset_name}_{modelparam_name}_k{n_clusters}_Z.npy',CKsym_Z)
    
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
    result_dir : str = 'results',
    aff_dir : str = 'results/affinity',
    pca_dim: int = 0
):
    #DATA HAS TO BE IN SHAPE OF N,D
    data = data.T
    data = min_max_norm(data)
    X_adj = DataProjection(data,pca_dim)
    print(X_adj.shape)
    B,Z = BDR_Solver(X_adj,modelparam,maxIter=maxIter)

    grps_B, CKsym_B = get_adj_grps(B,modelparam,n_clusters)
    grps_Z,CKsym_Z = get_adj_grps(Z,modelparam,n_clusters)
    #TO BE DONE
    if visualize:
        pass
    if save_affnt:save_affinities(CKsym_B,CKsym_Z,aff_dir,modelparam_name,dataset_name,n_clusters)

    #TO DO : SAVE EAVLUATION
    return grps_B,grps_Z
def get_adj_grps(aff,modelParam,n_clusters):
    CKsym , _ = BuildAdjacency(thrC(aff,modelParam.rho))
    grps = SpectralClustering(n_clusters=n_clusters,affinity='precomputed').fit_predict(CKsym)
    return grps, CKsym

def load_affnt(aff_dir):
    aff = np.load(aff_dir)
    return aff
def min_max_norm(A):
    min_val = np.min(A)
    max_val = np.max(A)

    # Perform min-max normalization
    normalized_matrix = (A - min_val) / (max_val - min_val)
    return A
