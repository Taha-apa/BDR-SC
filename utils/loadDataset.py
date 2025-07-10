from scipy.io import loadmat
import pickle
def load_dataset(data_dir):
    if ".mat" in data_dir:
        data = loadmat(data_dir)
        return data
    if ".pkl" in data_dir:
        with open(data_dir,'rb') as f:
            data = pickle.load(f)
            return data
    pass