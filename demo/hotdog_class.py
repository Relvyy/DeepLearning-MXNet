import mxnet.profiler

from dataset.hotdog import *
from tools.tools import *
from dataset.normalize import normalize as nm
from mxnet import nd, init, gluon
from mxnet.gluon import loss as gloss
from mxnet.gluon import data as gdata, model_zoo as mz
from tools.tools import _get_batch


def train(train_iter, test_iter, net, loss, trainer, ctx, num_epochs):
    print('Training On: ', ctx)
    if isinstance(ctx, mxnet.Context):
        ctx = [ctx]
    for epoch in range(num_epochs):
        train_l_sum, train_accu_sum, n, m, start = 0.0, 0.0, 0, 0, time.time()
        for i, batch in enumerate(train_iter):
            Xs, ys, batch_size = _get_batch(batch, ctx)
            ls = []
            with autograd.record():
                y_hats = [net(X) for X in Xs]
                ls = [loss(y_hat, y) for y_hat, y in zip(y_hats, ys)]
            for l in ls:
                l.backward()
            trainer.step(batch_size)
            train_l_sum += sum([l.sum().asscalar() for l in ls])
            n += sum([l.size for l in ls])
            train_accu_sum += sum([(y_hat.argmax(axis=1) == y).sum().asscalar() for y_hat, y in zip(y_hats, ys)])
            m += sum([y.size for y in ys])
        test_acc = evaluate_accuracy_gpus(test_iter, net, ctx)
        print('Epoch %d, Loss: %.4f, Train Acc %.3f, Test Acc: %.3f, Time: %.2f sec' % (
            epoch + 1, train_l_sum / n, train_accu_sum / m, test_acc, time.time() - start))



def train_finetune(net, lr, train_augs, test_augs, bs=128, num_epoch=5):
    train_imgs, test_imgs = get_hotdog()
    train_iter = gdata.DataLoader(train_imgs.transform_first(train_augs), bs, shuffle=True)
    test_iter = gdata.DataLoader(test_imgs.transform_first(test_augs), bs, shuffle=True)
    ctx = try_all_gpus()
    net.collect_params().reset_ctx(ctx)
    net.hybridize()
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr, 'wd': 0.001})
    train(train_iter, test_iter, net, loss, trainer, ctx, num_epoch)



def train_ft():
    root = os.path.join(os.getcwd(), '..\model')
    route = os.path.join(os.getcwd(), '..\data\hotdog\\train\hotdog')
    m, s = nm(route)
    normalize = gdata.vision.transforms.Normalize(m, s)

    train_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.RandomResizedCrop(224),
        gdata.vision.transforms.RandomFlipLeftRight(),
        gdata.vision.transforms.ToTensor(),
        normalize])

    test_augs = gdata.vision.transforms.Compose([
        gdata.vision.transforms.Resize(256),
        gdata.vision.transforms.CenterCrop(224),
        gdata.vision.transforms.ToTensor(),
        normalize])

    pretrained_net = mz.vision.resnet18_v2(root=root, pretrained=True)

    finetune_net = mz.vision.resnet18_v2(root=root, classes=2)
    finetune_net.features = pretrained_net.features
    finetune_net.output.initialize(init.Xavier())
    finetune_net.output.collect_params().setattr('lr_mult', 10)

    train_finetune(finetune_net, 0.01, train_augs, test_augs)
    print(finetune_net)


if __name__ == '__main__':
    train_ft()