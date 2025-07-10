from dataclasses import dataclass
import yaml
from pathlib import Path

@dataclass
class Args:
    dataset_name:str
    dataset_dir:str 
    affinity_dir : str
    out_dir : str
    visualize : bool
    save_aff : bool
    n_clusters : int
    maxIter: int
    modelParam_dir : str
    pca_dim : int
@dataclass
class synthDataArgs:
    N:int
    D:int
    n_original_clusters : int

def read_params(path_to : Path,modelClass):
    try:
        with open(path_to,'r') as f:
            f_content = f.read()
            params_dict = yaml.safe_load(f_content)
            ModelParams = modelClass(**params_dict)
            return ModelParams
    except OSError as e :
        print(f"file Not found : \n {e}")
        return None

