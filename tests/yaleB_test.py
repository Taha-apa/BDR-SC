from pathlib import Path
from model.BDR_Solver import modelConfig
from sklearn.datasets import make_blobs
from sklearn.metrics import f1_score
from scripts import BDR
from scripts.args import Args, read_params, synthDataArgs
from utils.loadDataset import load_dataset
if __name__ == "__main__" :
    data_dir = "./datasets/EYaleB_2016.mat"
    data_mat = load_dataset(data_dir)
    data_config_path = Path("./datasets/dataset_config/EYaleB_2016.yaml")
    data_args = read_params(data_config_path,Args)
    model_config_path = Path("./model/model_params/yaleB_config.yaml")
    X,y = data_mat['fea'].T, data_mat['gnd'].T
    ModelParams = read_params(model_config_path,modelConfig)
    grps_B,grps_Z = BDR.run(ModelParams,"yaleB","yaleB",data_args.maxIter,data_args.n_clusters,X,data_args.visualize,True,data_args.out_dir,)
    
    f_score_B= f1_score(y,grps_B,average='macro')
    print(f"f1 score for B : {f_score_B}")
    f_score_Z= f1_score(y,grps_Z,average='macro')
    print(f"f1 score for Z : {f_score_Z}")
