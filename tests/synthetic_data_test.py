from pathlib import Path
from model.BDR_Solver import modelConfig
from measure.Missclassification import missRate
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score
from scripts import BDR
from scripts.args import Args, read_params, synthDataArgs
import numpy as np

if __name__ == "__main__":
    dataArgs = synthDataArgs(1000,8,8)
    synth_config_path = Path("./datasets/dataset_config/synthDataset.yaml")
    
    args = read_params(synth_config_path,Args)
    randState = 42
    paramDir = Path(args.modelParam_dir)
    X,y = make_blobs(dataArgs.N,dataArgs.D,centers=dataArgs.n_original_clusters,random_state=randState)
    
    synthModelParams = read_params(paramDir,modelConfig)
    grps_B,grps_Z = BDR.run(synthModelParams,"synthTest",args.dataset_name,args.maxIter,args.n_clusters,X,False)
    f_score_B= f1_score(y,grps_B,average='macro')
    print(f"f1 score for B : {f_score_B}")
    f_score_Z= f1_score(y,grps_Z,average='macro')
    print(f"f1 score for Z : {f_score_Z}")