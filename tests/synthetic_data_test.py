from pathlib import Path
from model.BDR_Solver import modelConfig
from sklearn.datasets import make_blobs
from scripts import BDR
from scripts.args import Args, read_params, synthDataArgs


if __name__ == "__main__":
    dataArgs = synthDataArgs(100,3,3)
    synth_config_path = Path("./datasets/dataset_config/synthDataset.yaml")
    
    args = read_params(synth_config_path,Args)
    randState = 42
    paramDir = Path(args.modelParam_dir)
    synthData = make_blobs(dataArgs.N,dataArgs.D,centers=dataArgs.n_original_clusters,random_state=randState)
    print(synthData[0].shape)
    synthModelParams = read_params(paramDir,modelConfig)
    grps_B,grps_Z = BDR.run(synthModelParams,"synthTest",args.dataset_name,args.maxIter,args.n_clusters,synthData[0],False)

