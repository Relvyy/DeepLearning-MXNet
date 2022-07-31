import math
import numpy as np
from mxnet import nd


def gd(eta):
    x = 10
    results = [x]
    for i in range(10):
        x -= eta * 2 * x
        results.append(x)
    print('epoch 10, x:', x)
    return results


def train_2d(trainer):
    x1, x2, s1, s2 = -5, -2, 0, 0
    results = [(x1, x2)]
    for i in range(20):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('Epoch %d, x1 %f, x2 %f' % (i+1, x1, x2))
    return results


def f_2d(x1, x2):
    return x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2, eta=0.15):
    return x1 - eta * 2 * x1, x2 - eta * 4 * x2, 0, 0


def sgd_2d(x1, x2, s1, s2, eta=0.15):
    return x1 - eta * (2 * x1 + np.random.normal(0.1)), \
           x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0
