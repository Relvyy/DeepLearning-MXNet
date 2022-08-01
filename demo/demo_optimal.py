import time
from tools.optimal import *
from tools.loss import squared_loss, l2loss
from model.model_zoo import linreg
from tools.tools import show_loss
from dataset.airfoil_self_noise import get_data
from mxnet import nd, autograd, init, gluon
from mxnet.gluon import loss as gloss, data as gdata, nn


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


def train_gluon(trainer_name, hyperparams, features, labels, batch_size=10, num_epochs=2):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))

    loss = l2loss()

    def eval_loss():
        return loss(net(features), labels).mean().asscalar()

    ls = [eval_loss()]
    trainer = gluon.Trainer(net.collect_params(), trainer_name, hyperparams)
    data_iter = gdata.DataLoader(gdata.ArrayDataset(features, labels), batch_size, shuffle=True)
    for _ in range(num_epochs):
        start = time.time()
        for batch_i, (X, y) in enumerate(data_iter):
            with autograd.record():
                l = loss(net(X), y).mean()
            l.backward()
            trainer.step(batch_size)
            if(batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
    print('Loss: %f, %f sec per epoch' % (ls[-1], time.time() - start))
    show_loss(num_epochs, ls)


if __name__ == '__main__':
    lr = 50
    num_epochs = 200
    batch_size = 1500
    features, labels = get_data()
    #train(sgd, None, {'lr': lr}, features, labels, batch_size, num_epochs)
    #train_gluon('sgd', {'learning_rate': lr}, features, labels, batch_size, num_epochs)
    #train(sgd_momentum, init_momentum_states(features),
    #      {'learning_rate': 0.004, 'momentum': 0.9}, features, labels)
    train_gluon('sgd', {'learning_rate': 0.004, 'momentum': 0.9}, features, labels)
