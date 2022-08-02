import os
import sys
import time
import numpy as np
import mxnet as mx
from IPython import display
from matplotlib import pyplot as plt
from mxnet.gluon import nn, loss as gloss, data as gdata
from mxnet import nd, autograd, gluon, init


class Tools():
    def __init__(self, **kwargs):
        super(Tools).__init__(**kwargs)
        self.batch_size = 256

    def try_gpu(self):
        try:
            if mx.__version__.find('gpu'):
                ctx = mx.gpu()
                _ = nd.array((1,), ctx=ctx)
            else:
                raise Exception('The Machine don''t  have gpu!')
        except mx.base.MXNetError:
            ctx = mx.cpu()
        return ctx

    def evaluate_accuracy(self, net, data_iter, ctx):
        accu_sum, n = nd.array([0], ctx=ctx), 0
        for X, y in data_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
            accu_sum += (net(X).argmax(axis=1) == y).sum()
            n += y.size
        return accu_sum.asscalar() / n

    def load_data_fashion_mnist(self, resize):  # , root=os.path.join('~', '.mxnet', 'datasets', 'fashion-mnist')):
        r = os.getcwd()
        root = os.path.join(r, 'data', 'fashion-mnist')
        root = os.path.expanduser(root)
        transformer = []
        if resize:
            transformer += [gdata.vision.transforms.Resize(resize)]
        transformer += [gdata.vision.transforms.ToTensor()]
        transformer = gdata.vision.transforms.Compose(transformer)
        mnist_train = gdata.vision.FashionMNIST(root=root, train=True)
        mnist_test = gdata.vision.FashionMNIST(root=root, train=False)
        num_workers = 0 if sys.platform.endswith('win') else 4
        train_iter = gdata.DataLoader(mnist_train.transform_first(transformer), self.batch_size, shuffle=True,
                                      num_workers=num_workers)
        test_iter = gdata.DataLoader(mnist_test.transform_first(transformer), self.batch_size, shuffle=True,
                                     num_workers=num_workers)

        return train_iter, test_iter

    def get_fashion_mnist_labels(self, labels):
        text_labels = ['T-Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                       'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
        return [text_labels[int(i)] for i in labels]

    def show_fashion_mnist(self, images, labels, colors, r):
        _, figs = plt.subplots(1, len(images), figsize=(12, 12))
        for f, img, lab, col in zip(figs, images, labels, colors):
            f.imshow(img.reshape((r, r)).asnumpy())
            f.set_title(lab, color=col)
            f.axes.get_xaxis().set_visible(False)
            f.axes.get_yaxis().set_visible(False)

    def test(self, net, test_iter, r):
        for X, y in test_iter:
            true_labels = self.get_fashion_mnist_labels(y.asnumpy())
            pred_labels = self.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())

            titles = ['T:' + true + '\n' + 'P:' + pred for true, pred in zip(true_labels, pred_labels)]
            color = ['g' if pred == true else 'r' for true, pred in zip(true_labels, pred_labels)]
            self.show_fashion_mnist(X[0:9], titles[0:9], color[0:9], r)
            break
        plt.show()

    def save(self, net, mode_str, name):
        if not os.path.exists('runs'):
            os.makedirs('runs')
        filename = 'runs\\' + mode_str + '_' + str(name) + '.params'
        if os.path.exists(filename):
            os.remove(filename)
        net.save_parameters(filename)

    def download_cifar(self):  # , root=os.path.join('~', '.mxnet', 'datasets', 'fashion-mnist')):
        r = os.getcwd()
        root = os.path.join(r, 'data', 'cifar-10')
        root = os.path.expanduser(root)
        cifar10_train = gdata.vision.CIFAR10(root=root, train=True)
        cifar10_test = gdata.vision.CIFAR10(root=root, train=False)

        return cifar10_train, cifar10_test


def show_trace(res):
    n = max(abs(min(res)), abs(max(res)), 10)
    f_line = np.arange(-n, n, 0.1)
    display.set_matplotlib_formats('svg')
    plt.rcParams['figure.figsize'] = (3.5, 2.5)
    plt.plot(f_line, [x * x for x in f_line])
    plt.plot(res, [x * x for x in res], '-o')
    plt.xlabel('x')
    plt.ylabel('f(x)')

    plt.show()


def show_trace_2d(f, results):
    plt.plot(*zip(*results), '-o', color='#ff7f0e')
    x1, x2 = np.meshgrid(np.arange(-10, 10, 0.05), np.arange(-5.0, 5.0, 0.05))
    plt.contour(x1, x2, f(x1, x2), colors='#1f77b4')
    plt.xlabel('x1')
    plt.ylabel('x2')

    plt.show()


def show_loss(epoch, ls):
    plt.rcParams['figure.figsize'] = (5, 3.5)
    plt.plot(np.linspace(0, epoch, len(ls)), ls)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()

    plt.show()


def show_image(imgs, rows, cols, scale=2):
    figsize = (cols * scale, rows * scale)
    _, axes = plt.subplots(rows, cols, figsize=figsize)
    for i in range(rows):
        for j in range(cols):
            axes[i][j].imshow(imgs[i * cols + j].asnumpy())
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
            axes[i][j].spines['right'].set_visible(False)
            axes[i][j].spines['top'].set_visible(False)
            axes[i][j].spines['bottom'].set_visible(False)
            axes[i][j].spines['left'].set_visible(False)

    plt.axis('off')
    plt.show()
