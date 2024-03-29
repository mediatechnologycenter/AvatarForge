import torch
import cv2
import numpy as np
import os


def load_dir(lmspath, framepath, start, end):
    lmss = []
    imgs_paths = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(lmspath, '%05d.lms' % i)):
            lms = np.loadtxt(os.path.join(
                lmspath, '%05d.lms' % i), dtype=np.float32)
            lmss.append(lms)
            imgs_paths.append(os.path.join(framepath, '%05d.jpg' % i))
    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).cuda()
    return lmss, imgs_paths
