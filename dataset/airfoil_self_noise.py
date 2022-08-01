import numpy as np
from mxnet import nd


def get_data():
    data = np.genfromtxt(r'..\data\airfoil_self_noise.dat', delimiter='\t')
    data = (data - data.mean(axis=0)) / data.std(axis=0)

    return nd.array(data[:1500, :-1]), nd.array(data[:1500, -1])
