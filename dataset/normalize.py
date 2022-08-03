import os
import cv2
import numpy as np

def normalize(route):
    rm, gm, bm = [], [], []
    rs, gs, bs = [], [], []
    file_list = os.listdir(route)
    for file in file_list:
        f = os.path.join(os.path.abspath(route), file)
        img = cv2.imread(f, 1)
        rm.append(np.mean(img[:, :, 0] / 255))
        gm.append(np.mean(img[:, :, 1] / 255))
        bm.append(np.mean(img[:, :, 2] / 255))

        rs.append(np.std(img[:, :, 0] / 255))
        gs.append(np.std(img[:, :, 1] / 255))
        bs.append(np.std(img[:, :, 2] / 255))

    r_mean = np.around(np.mean(rm), 3)
    g_mean = np.around(np.mean(gm), 3)
    b_mean = np.around(np.mean(bm), 3)

    r_std = np.around(np.std(rs), 3)
    g_std = np.around(np.std(gs), 3)
    b_std = np.around(np.std(bs), 3)

    mean = [r_mean, g_mean, b_mean]
    std = [r_std, g_std, b_std]

    return mean, std

