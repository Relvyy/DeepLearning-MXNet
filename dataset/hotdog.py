import os
import zipfile
from mxnet.gluon import utils as gutils, data as gdata

def get_hotdog():
    data_dir = os.path.join(os.getcwd(), '..\data\\')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    base_url = 'https://apache-mxnet.s3-accelerate.amazonaws.com/'
    fname = gutils.download(base_url+'gluon/dataset/hotdog.zip', path=data_dir,
                            sha1_hash='fba480ffa8aa7e0febbb511d181409f899b9baa5')
    with zipfile.ZipFile(fname, 'r') as z:
        z.extractall(data_dir)

    # 读取当前父文件夹下的所有子文件夹的所有文件
    train_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'hotdog\\train'))
    test_imgs = gdata.vision.ImageFolderDataset(os.path.join(data_dir, 'hotdog\\test'))

    return train_imgs, test_imgs

