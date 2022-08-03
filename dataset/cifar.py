import os
import sys
from mxnet.gluon import data as gdata

cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

cifar100_corse_labels = ['mammals', 'fish', 'flowers', 'containers', 'fruit & vegetbles',
                        'electrical', 'furniture', 'insects', 'carnivores',
                        'manmad', 'outdoor_scenes', 'omnivores & herbivores',
                        'medium_mammals', 'invertebrates', 'people', 'reptiles', 'small_mammals',
                        'trees', 'vehicles_1', 'vehicles_2']

cifar100_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed',
                   'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge',
                   'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar',
                   'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach',
                   'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
                   'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
                   'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard',
                   'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle',
                   'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                   'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain',
                   'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon',
                   'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
                   'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel',
                   'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
                   'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle',
                   'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm', ]

def cifar10(batch_size=10, resize=(32,32)):
    r = os.getcwd()
    root = os.path.join(r, 'data', 'cifar-10')
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    cifar10_train = gdata.vision.CIFAR10(root=root, train=True)
    cifar10_test = gdata.vision.CIFAR10(root=root, train=False)
    '''
    num_workers = 0 if sys.platform.endswith('win') else 4
    train_iter = gdata.DataLoader(cifar10_train.transform_first(transformer), batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(cifar10_test.transform_first(transformer), batch_size, shuffle=True,
                                 num_workers=num_workers)
    '''

    return cifar10_train, cifar10_test


def cifar100(batch_size=10, resize=(32, 32)):
    r = os.getcwd()
    root = os.path.join(r, 'data', 'cifar-100')
    root = os.path.expanduser(root)
    transformer = []
    if resize:
        transformer += [gdata.vision.transforms.Resize(resize)]
    transformer += [gdata.vision.transforms.ToTensor()]
    transformer = gdata.vision.transforms.Compose(transformer)
    cifar100_train = gdata.vision.CIFAR100(root=root, train=True)
    cifar100_test = gdata.vision.CIFAR100(root=root, train=False)
    '''
    num_workers = 0 if sys.platform.endswith('win') else 4
    train_iter = gdata.DataLoader(cifar100_train.transform_first(transformer), batch_size, shuffle=True,
                                  num_workers=num_workers)
    test_iter = gdata.DataLoader(cifar100_test.transform_first(transformer), batch_size, shuffle=True,
                                 num_workers=num_workers)
    '''

    return cifar100_train, cifar100_test


def label2txt(imgs, cls='cifar10'):
    txt = []
    r = []
    if cls == 'cifar10':
        labels = cifar10_labels
    elif cls == 'fine':
        labels = cifar100_corse_labels
    else:
        labels = cifar100_corse_labels
    for i in imgs[1]:
        r.append(i)
        txt.append(labels[i])
    return txt

def txtop():
    route = os.path.join(os.getcwd(), 'data\cifar-100\labels.txt')
    print(route)
    with open(route, 'r+') as f:
        line = f.readlines()
        print(len(line))
        s = ''
        for i in range(len(line)):
            s += '\''+line[i]
        print(s)
        new = s.replace('\n', '\', ')
        print(new)
        f.write(new)
    f.close()
