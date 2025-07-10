import numpy as np
import re
import pickle
usps_dir = './datasets/usps'
with open(usps_dir,"r") as file:
    usps_raw = file.read()
    usps = usps_raw.split('\n')
    N = len(usps)
    D = 256
    X = np.zeros((N,D))
    y = np.zeros((N,)) 
    
    for i in range(0,N -1 ):
        y[i] = int(usps[i][0])
        pattern = r":\s*(-?\d+\.?\d*)"
        floats = re.findall(pattern,usps[i])
        X[i] = [float(fs) for fs in floats]
    data = dict({
        'X' : X,
        'y': y
    })
    output_dir = "./datasets/usps.pkl"
    with open(output_dir,"wb") as f:
        pickle.dump(data,f)
    