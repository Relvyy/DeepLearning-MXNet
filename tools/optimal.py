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
    for i in range(200):
        x1, x2, s1, s2 = trainer(x1, x2, s1, s2)
        results.append((x1, x2))
    print('Epoch %d, x1 %f, x2 %f' % (i+1, x1, x2))
    return results


def f_2d(x1, x2):
    return 0.01 * x1 ** 2 + 2 * x2 ** 2


def gd_2d(x1, x2, s1, s2, eta=0.3):
    return x1 - eta * 0.02 * x1, x2 - eta * 4 * x2, 0, 0


def sgd_2d(x1, x2, s1, s2, eta=0.15):
    return x1 - eta * (2 * x1 + np.random.normal(0.1)), \
           x2 - eta * (4 * x2 + np.random.normal(0.1)), 0, 0


def sgd(params, states, hyperparams):
    for p in params:
        p[:] -= hyperparams['lr'] * p.grad


def momentum(x1, x2, v1, v2, eta=0.4, gamma=0.9):
    v1 = gamma * v1 + eta * 0.02 * x1
    v2 = gamma * v2 + eta * 4 * x2
    return x1 - v1, x2 - v2, v1, v2


def init_momentum_states(features):
    v_w = nd.zeros((features.shape[1], 1))
    v_b = nd.zeros(1)
    return v_w, v_b


def sgd_momentum(params, states, hyperparams):
    for p, v in zip(params, states):
        v[:] = hyperparams['momentum'] * v + hyperparams['learning_rate'] * p.grad
        p[:] -= v
