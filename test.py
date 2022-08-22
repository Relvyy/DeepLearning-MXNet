from tools.tools import *
from tools.optimal import *
from dataset.cifar import *
from dataset.hotdog import *
from model.model_zoo import *
from mxnet import image, contrib
from demo.hotdog_class import model as m


def test(model_string, r=224):
    models = Model()
    tool = Tools()
    tool.try_gpu()

    #model = models.model(model_string)
    #model.initialize()
    model, _, test_augs = m()
    train_imgs, test_imgs = get_hotdog()
    test_iter = gdata.DataLoader(test_imgs.transform_first(test_augs), batch_size=128, shuffle=True)
    model.load_parameters(f'.\demo\\runs\\{model_string}_best.params')
    #_, test_iter = tool.load_data_fashion_mnist(resize=r)
    tool.test_hotdog(model, test_iter)


if __name__ == '__main__':
    #test('ResNet', 224)
    #show_trace(gd(0.9))
    #show_trace_2d(f_2d, train_2d(momentum))
    #show_trace_2d(f_2d, train_2d(sgd_2d))
    #txtop()
    '''
    route = os.path.join(os.getcwd(), 'data\pictures\cat3.jpg')
    img = image.imread(route)
    aug1 = gdata.vision.transforms.RandomColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5)
    aug2 = gdata.vision.transforms.RandomResizedCrop((200, 200), ratio=(0.5, 2), scale=(0.1, 1))
    aug3 = gdata.vision.transforms.RandomFlipLeftRight()
    aug4 = gdata.vision.transforms.RandomFlipTopBottom()
    augs = gdata.vision.transforms.Compose([aug1, aug2, aug3, aug4])
    aug_apply(img, augs)
    

    train, test = get_hotdog()
    print(len(train))
    h = [train[i][0] for i in range(8)]
    n = [train[-i - 1][0] for i in range(8)]
    show_image(h+n, [], 2, 8)
    
    print(try_all_gpus())
    test('finetune_resnet18v2')
    '''
    img = image.imread(r'd:\Documents\DeeplLearning\DeepLearning-MXNet\data\pictures\catdog.jpg').asnumpy()
    fig = plt.imshow(img)
    h, w = img.shape[0:2]
    X = nd.random.uniform(shape=(1, 3, w, h))
    Y = contrib.nd.MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
    boxes = Y.reshape(h, w, 5, 4)

    scale = nd.array((w, h, w, h))
    for i in range(30, 36):
        for j in range(30, 36):
            show_bboxes(fig.axes, boxes[i, j, :, :]*scale, ['s=0.75, r=1', 's=0.5, r=1', 's=0.25, r=1',
                                                        's=0.75, r=2', 's=0.75, r=0.5'])

    plt.show()

