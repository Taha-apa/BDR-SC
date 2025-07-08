from scipy.io import loadmat
def load_dataset(data_dir):
    if ".mat" in data_dir:
        data = loadmat(data_dir)
        return data
    pass