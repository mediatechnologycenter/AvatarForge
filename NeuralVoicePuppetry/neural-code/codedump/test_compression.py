# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
from subprocess import call
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.metrics import structural_similarity


def check_mkdir(path):
    if not os.path.exists(path):
        print('creating %s' % path)
        os.makedirs(path)


def video2sequence_lossless(video_path):
    videofolder = video_path.split('.')[0]
    check_mkdir(videofolder)
    video_name = video_path.split('/')[-1].split('.')[0]
    cmd = (f'ffmpeg -i {video_path} -vf fps=25 {videofolder}/{video_name}_frame%04d.png').split()
    call(cmd)
    imagepath_list = [os.path.join(videofolder, f) for f in os.listdir(videofolder)]
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list


def compute_diff(image1, image2, text):
    # Convert images to grayscale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(image1, image2, full=True)
    print(text, ' SSIM: ', score)
    diff = (diff * 255).astype("uint8")

    plt.imshow(diff)
    plt.show()


NAME = 'DynamicNeuralTextures-Demo_Female_moderator_1'
FILE_ID_SOURCE = 'Clara_audios_Resemble_clara'
FILE_ID_TARGET = 'Demo_Female_moderator_1'

video_out_path = f'results/inference/{NAME}/{FILE_ID_SOURCE}_to_{FILE_ID_TARGET}/'
video_fname = os.path.join(video_out_path, f'{FILE_ID_SOURCE}_to_{FILE_ID_TARGET}.mp4')
frames_path = '/home/alberto/data/videosynth/External/Demo/Female_moderator_1/'

imagepath_list = video2sequence_lossless(video_fname)

for gen_frame, old_frame in zip(tqdm(sorted(imagepath_list)), sorted(os.listdir(frames_path))):

    old_image = cv2.imread(os.path.join(frames_path, old_frame))
    gen_image = cv2.imread(gen_frame)

    compute_diff(old_image, gen_image, 'old_image - gen_image')

