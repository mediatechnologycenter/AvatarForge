# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform, warp, resize, rescale
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third/DECA/')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg


def warp_back(image, oldimage, tform):

    alpha = 0.6

    oldimage = oldimage.astype(np.float64) /255.
    new_size = oldimage.shape

    dst_image = warp(image, tform, output_shape=new_size)

    # Mask of non-black pixels.
    mask = np.where(np.all(dst_image == [0, 0, 0], axis=-1))
    dst_image[mask] = oldimage[mask]

    res = cv2.addWeighted(oldimage, 1 - alpha, dst_image, alpha, 0)
    res = res[:, :, ::-1]

    return res


class DECA_tracker:
    def __init__(self, video_path, target_dir=None):

        # load test images
        self.testdata = datasets.TestData(video_path, iscrop=True, face_detector='fan', target_dir=target_dir)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # run DECA
        deca_cfg.model.use_tex = False
        self.deca = DECA(config=deca_cfg, device=self.device)

    def __call__(self, images, tform=None):

        codedict = self.deca.encode(images)

        opdict, visdict = self.deca.decode(codedict, tform)

        mask = self.deca.render_mask(opdict['grid'])

        # for key in opdict.keys():
        #     print(key, opdict[key].size())

        return codedict, opdict, mask
