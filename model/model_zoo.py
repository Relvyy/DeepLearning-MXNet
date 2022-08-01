from mxnet import nd, autograd, init, gluon
from mxnet.gluon import data as gdata, loss as gloss, nn


class Inception(nn.Block):
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        self.p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        self.p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1, activation='relu')
        self.p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2, activation='relu')
        self.p4_1 = nn.MaxPool2D(pool_size=3, strides=1, padding=1)
        self.p4_2 = nn.Conv2D(c4, kernel_size=1, activation='relu')

    def forward(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))

        return nd.concat(p1, p2, p3, p4, dim=1)
class Residual(nn.Block):
    def __init__(self, num_channels, use_1x1_conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = nn.Conv2D(num_channels, kernel_size=3, padding=1, strides=strides)
        self.conv2 = nn.Conv2D(num_channels, kernel_size=3, padding=1)
        if use_1x1_conv:
            self.conv3 = nn.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None

        self.bn = nn.BatchNorm()

    def forward(self, X):
        Y = nd.relu(self.bn(self.conv1(X)))
        Y = self.bn(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return nd.relu(Y + X)

class DenseBlock(nn.Block):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.net = nn.Sequential()
        for _ in range(num_convs):
            self.net.add(self.conv_block(num_channels))


    def conv_block(self, num_channels):
        cbl = nn.Sequential()
        cbl.add(nn.BatchNorm(),
                nn.Activation('relu'),
                nn.Conv2D(num_channels, kernel_size=3, padding=1))

        return cbl

    def transition_block(self, num_channels):
        tbl = nn.Sequential()
        tbl.add(nn.BatchNorm(),nn.Activation('relu'),
                nn.Conv2D(num_channels, kernel_size=1),
                nn.AvgPool2D(pool_size=2, strides=2))
        return tbl

    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            X = nd.concat(X, Y, dim=1)

        return X

class Model(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)

    def lenet(self):
        net = nn.Sequential()
        net.add(nn.Conv2D(6, kernel_size=5, activation='sigmoid'),
                nn.MaxPool2D(pool_size=2, strides=2),
                nn.Conv2D(16, kernel_size=5, activation='sigmoid'),
                nn.MaxPool2D(pool_size=1, strides=2),
                nn.Dense(120, activation='sigmoid'),
                nn.Dense(80, activation='sigmoid'),
                nn.Dense(10))

        return net

    def alexnet(self):
        net = nn.Sequential()
        net.add(nn.Conv2D(96, kernel_size=11, strides=4, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(256, kernel_size=5, padding=2, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(384, kernel_size=3, padding=1, activation='relu'),
                nn.Conv2D(256, kernel_size=3, padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2),
                nn.Dense(4096, activation='relu'),
                nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'),
                nn.Dropout(0.5),
                nn.Dense(10))
        return net
    def vgg_block(self, num_convs, num_channels):
        blk = nn.Sequential()
        for _ in range(num_convs):
            blk.add(nn.Conv2D(num_channels, kernel_size=3, padding=1, activation='relu'))
        blk.add(nn.MaxPool2D(pool_size=2, strides=2))

        return blk

    def vgg(self):
        conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))
        net = nn.Sequential()
        for (num_convs, num_channels) in conv_arch:
            net.add(self.vgg_block(num_convs, num_channels))

        net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
                nn.Dense(10))

        return net
    def nin_block(self, num_channels, kernel_size, strides, padding):
        nbl = nn.Sequential()
        nbl.add(nn.Conv2D(num_channels, kernel_size, strides, padding, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'),
                nn.Conv2D(num_channels, kernel_size=1, activation='relu'))
        return nbl

    def nin(self):
        net = nn.Sequential()
        net.add(nn.Conv2D(96, kernel_size=11, strides=4, padding=0),
                nn.MaxPool2D(pool_size=3, strides=2),
                self.nin_block(256, kernel_size=5, strides=1, padding=2),
                nn.MaxPool2D(pool_size=3, strides=2),
                self.nin_block(384, kernel_size=3, strides=1, padding=1),
                nn.MaxPool2D(pool_size=3, strides=1),
                nn.Dropout(0.5),
                self.nin_block(10, kernel_size=3, strides=1, padding=1),
                nn.GlobalAvgPool2D(),
                nn.Flatten())

        return net
    def inception(self, c1, c2, c3, c4):
        p1_1 = nn.Conv2D(c1, kernel_size=1, activation='relu')
        p2_1 = nn.Conv2D(c2[0], kernel_size=1, activation='relu')
        p2_2 = nn.Conv2D(c2[1], kernel_size=3, padding=1, activation='relu')
        p3_1 = nn.Conv2D(c3[0], kernel_size=1, activation='relu')
        p3_2 = nn.Conv2D(c3[1], kernel_size=5, padding=2, activation='relu')
        p4_1 = nn.Conv2D(c4[0], kernel_size=3, strides=1, padding=1)
        p4_2 = nn.Conv2D(c4[1], kernel_size=1, activation='relu')

        p1 = p1_1()
        p2 = p2_2(p2_1())
        p3 = p3_2(p3_1())
        p4 = p4_2(p4_1())

        return nd.concat(p1, p2, p3, p4, dim=1)
    def googlenet(self):
        net = nn.Sequential()
        b1 = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1),
                nn.Conv2D(64, kernel_size=1, activation='relu'),
                nn.Conv2D(192, kernel_size=3, padding=1, activation='relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1),
                Inception(64, (96, 128), (16, 32), 32),
                Inception(128, (128, 193), (32, 96), 64),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1),
                Inception(192, (96, 208), (16, 48), 64),
                Inception(160, (112, 224), (24, 64), 64),
                Inception(128, (128, 256),(24, 64), 64),
                Inception(112, (144, 288),(32, 64), 64),
                Inception(256, (160, 320), (32, 128), 128),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1),
                Inception(256, (160, 320), (32, 128), 128),
                Inception(384, (192, 384), (48, 128), 128),
                nn.GlobalAvgPool2D(),
                nn.Dense(10)
                )

        return net
    def resnet_block(self, num_channels, num_residuals, first_block=False):
        rbl = nn.Sequential()
        for i in range(num_residuals):
            if i == 0 and not first_block:
                rbl.add(Residual(num_channels, use_1x1_conv=True, strides=2))
            else:
                rbl.add(Residual(num_channels))
        return rbl

    def resnet(self):
        net = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                nn.BatchNorm(), nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1),
                self.resnet_block(64, 2, first_block=True),
                self.resnet_block(128, 2),
                self.resnet_block(256, 2),
                self.resnet_block(512, 2),
                nn.GlobalAvgPool2D(),
                nn.Dense(10))

        return net
    def transition_block(self, num_channels):
        tbl = nn.Sequential()
        tbl.add(nn.BatchNorm(), nn.Activation('relu'),
                nn.Conv2D(num_channels, kernel_size=1),
                nn.AvgPool2D(pool_size=2, strides=2))
        return tbl

    def densenet(self):
        net = nn.Sequential()
        net.add(nn.Conv2D(64, kernel_size=7, strides=2, padding=3),
                nn.BatchNorm(), nn.Activation('relu'),
                nn.MaxPool2D(pool_size=3, strides=2, padding=1))
        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]

        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            net.add(DenseBlock(num_convs, growth_rate))
            num_channels += num_convs * growth_rate
            if i != len(num_convs_in_dense_blocks) - 1:
                num_channels //= 2
                net.add(self.transition_block(num_channels))
        net.add(nn.BatchNorm(), nn.Activation('relu'),
                nn.GlobalAvgPool2D(),
                nn.Dense(10))

        return net
    def test(self, net, r):
        x = nd.random.uniform(shape=(1, 1, r, r))
        net.initialize()
        for layer in net:
            x = layer(x)
            print(layer.name, ': ', 'Output Shape:\t', x.shape)

    def model(self, model_string):
        model_string = model_string.lower()
        model_dict = {'lenet': self.lenet(), 'alexnet': self.alexnet(), 'vgg': self.vgg(),
                      'nin': self.nin(), 'googlenet': self.googlenet(),
                      'resnet': self.resnet(), 'densenet': self.densenet()}
        return model_dict[model_string]


def linreg(X, w, b):
    return nd.dot(X, w) + b

