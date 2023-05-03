# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
from psbody.mesh import Mesh
import cv2
import pyrender
import trimesh
import tempfile
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from psbody.mesh import Mesh
import tempfile
from subprocess import call
from tqdm import tqdm
from PIL import Image


def make_video_seq(audio_fname, images_out_path, video_out_path, name):

    image = cv2.imread(os.path.join(images_out_path, sorted(os.listdir(images_out_path))[0]))
    size = (image.shape[1], image.shape[0])
    print('Video size: ', size)

    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=video_out_path)
    if int(cv2.__version__[0]) < 3:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.cv.CV_FOURCC(*'mp4v'), 25, size, True)
    else:
        writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), 25, size, True)

    images_list = sorted(os.listdir(images_out_path))
    indexes = []

    for i in images_list:
        if i.startswith('.'):
            images_list.remove(i)

        else:
            indexes.append(int(i[:-4]))

    print(np.max(indexes))

    for i in tqdm(range(np.max(indexes))):

        file_name = os.path.join(images_out_path, '%04d.jpg' % i)
        im = cv2.imread(file_name)
        writer.write(im)

    writer.release()

    video_fname = os.path.join(video_out_path, name)
    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -pix_fmt yuv420p {2}'.format(
        audio_fname, tmp_video_file.name, video_fname)).split()
    call(cmd)


if __name__ == '__main__':
    audio_fname = '/home/alberto/data/videosynth/SRF_anchor_short/Halbtotale/355_9415.wav'
    video_out_path = 'results/face_reconstruction/video'
    os.makedirs(video_out_path, exist_ok=True)
    # images_out_path = 'results/face_reconstruction/images'
    images_out_path = 'results/face_reconstruction/combined_images'
    os.makedirs(images_out_path, exist_ok=True)

    make_video_seq(audio_fname, images_out_path, video_out_path)
