from mxnet import nd
from mxnet.gluon import loss as gloss


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


def l2loss():
    return gloss.L2Loss()


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()
