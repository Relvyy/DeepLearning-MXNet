import time
from tools.optimal import sgd
from tools.loss import squared_loss
from model.model_zoo import linreg
from tools.tools import show_loss
from dataset.airfoil_self_noise import get_data
from mxnet import nd, autograd, init, gluon
from mxnet.gluon import loss as gloss, data as gdata


def train(train_fn, states, hyperparams, features, labels, batch_size=10, num_epochs=2):
    net, loss = linreg, squared_loss
    w = nd.random.normal(scale=0.01, shape=(features.shape[1], 1))
    b = nd.zeros(1)
    w.attach_grad()
    b.attach_grad()

    def eval_loss():
        return loss(net(features, w,  b), labels).mean().asscalar()

    ls = [eval_loss()]
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X, w, b), y).mean()
            l.backward()
            train_fn([w, b], states, hyperparams)
            if(batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('Loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    show_loss(num_epochs, ls)


if __name__ == '__main__':
    lr = 0.05
    num_epochs = 6
    batch_size = 10
    features, labels = get_data()
    train(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)
