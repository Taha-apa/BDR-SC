import argparse
import sys

from dataclasses import dataclass
from model.BDR_Solver import modelConfig
from sklearn.datasets import make_blobs
import yaml
import os
from pathlib import Path
import BDR
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
        print(f"file Not found \n {e}")
        return None

if __name__ == "__main__":
    dataArgs = synthDataArgs(100,3,3)
    synth_config_path = Path("./datasets/dataset_config/synthDataset.yaml")
    args = read_params(synth_config_path,Args)
    randState = 42
    paramDir = Path(args.modelParam_dir)
    synthData = make_blobs(dataArgs.N,dataArgs.D,centers=dataArgs.n_original_clusters,random_state=randState)
    synthModelParams = read_params(paramDir,modelConfig)
    grps_B,grps_Z = BDR.run(synthModelParams,"synthTest",**args)
