from model.model_zoo import *
from tools.tools import *


class Train():
    def __init__(self, net, batch_size, num_epochs, ctx, **kwargs):
        super(Train).__init__(**kwargs)
        self.net = net
        self.ctx = ctx
        self.batch_size = batch_size
        self.num_epochs = num_epochs

    def train_ch5(self, train_iter, test_iter, trainer, mode_str, detail=False):
        print('Training On', self.ctx)
        loss = gloss.SoftmaxCrossEntropyLoss()
        optm = 1e10
        for epoch in range(num_epochs):
            train_l_sum, train_accu_sum, n, start = 0.0, 0.0, 0, time.time()
            for i, (X, y) in enumerate(train_iter):
                tm = time.time()
                X, y = X.as_in_context(self.ctx), y.as_in_context(self.ctx)
                with autograd.record():
                    y_hat = net(X)
                    l = loss(y_hat, y).sum()
                l.backward()
                trainer.step(self.batch_size)
                y = y.astype('float32')
                train_l_sum += l.asscalar()
                train_accu_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
                n += y.size
                if detail:
                    ct = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                    print('%s %2d: %3d/%3d time: %.3f sec' % (ct, epoch + 1, i + 1, len(train_iter), time.time() - tm))
            test_accu = tool.evaluate_accuracy(net, test_iter, ctx)
            if test_accu <= optm:
                tool.save(net, mode_str, 'best')
            if epoch == num_epochs - 1:
                tool.save(net, mode_str, 'last')
            print('Epoch %d, Loss %.4f, Train Acc %.3f, Test Acc %.3f, Time %.3f sec' % (epoch + 1,
                                                                                         train_l_sum / n,
                                                                                         train_accu_sum / n, test_accu,
                                                                                         time.time() - start))
        return net


if __name__ == '__main__':
    model_string = 'ResNet'
    model = Model()
    tool = Tools()

    r = 224
    lr, batch_size, num_epochs, ctx = 0.05, 256, 10, tool.try_gpu()
    net = model.model(model_string)
    net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
    net.hybridize()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    train_iter, test_iter = tool.load_data_fashion_mnist(resize=r)

    train = Train(net, batch_size, num_epochs, ctx)
    net = train.train_ch5(train_iter, test_iter, trainer, model_string, detail=True)
    #tool.test(net, test_iter, r)
