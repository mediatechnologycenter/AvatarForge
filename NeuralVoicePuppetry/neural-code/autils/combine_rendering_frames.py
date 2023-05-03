# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import warp
from skimage.transform import SimilarityTransform

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

images_out_path = 'results/face_reconstruction/images'
os.makedirs(images_out_path, exist_ok=True)

tform_path = '/home/alberto/NeuralVoicePuppetry/datasets/SRF_anchor_short/Halbtotale_355_9415/tform.npy'

frames_path = '/home/alberto/data/videosynth/SRF_anchor_short/Halbtotale/355_9415'

combined_images_out_path = 'results/face_reconstruction/combines_images'
os.makedirs(combined_images_out_path, exist_ok=True)

tform = np.load(tform_path)

for image_path, frame in zip(tqdm(sorted(os.listdir(images_out_path))), sorted(os.listdir(frames_path))):

    index = int(image_path[:-4])
    f_tform = SimilarityTransform(matrix=tform[index])

    image = cv2.imread(os.path.join(images_out_path,image_path))
    old_image = cv2.imread(os.path.join(frames_path, frame))

    res = warp_back(image, old_image, f_tform)
    res = res * 255.
    res = res.astype('uint8')
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)


    file_name = os.path.join(combined_images_out_path, '%04d.jpg' % index)
    cv2.imwrite(file_name, res)


