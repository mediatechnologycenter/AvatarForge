# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import shutil
import subprocess

import cv2
import h5py
import numpy as np
from scipy.ndimage.morphology import binary_erosion
from skimage.metrics import structural_similarity
from skimage.transform import SimilarityTransform
from skimage.transform import warp
from tqdm import tqdm

from base_options import PostprocessingOptions


def warp_back(image, oldimage, tform):
    alpha = 1
    new_size = oldimage.shape

    dst_image = (warp(image, tform, output_shape=new_size) * 255.).astype(np.uint8)

    # Mask of non-black pixels.
    ones = np.ones_like(dst_image[:, :, 0]).astype(np.uint8)
    indexes = np.where(np.all(dst_image == [0, 0, 0], axis=-1))
    ones[indexes] = 0
    eroded_mask = binary_erosion(ones, iterations=5).astype(np.uint8)

    dst_image[eroded_mask == 0] = oldimage[eroded_mask == 0]

    return dst_image


def compute_diff(image1, image2, text):
    # Convert images to grayscale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(image1, image2, full=True)
    print(text, ' SSIM: ', score)
    diff = (diff * 255).astype("uint8")


def write_video_with_audio(save_root, audio_path, output_path, h=512, w=512, fps=25):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1.0
    fontColor = (255, 255, 255)
    thickness = 1
    lineType = 2
    text = 'This video has been manipulated.'

    label_width, label_height = cv2.getTextSize(text, font, 1, 2)[0]
    print(label_width, label_height)
    bottomLeftCornerOfText = (int((w - label_width) / 2), int(h - label_height - 20))

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    # fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_tmp_path = os.path.join(save_root, 'tmp.avi')
    out = cv2.VideoWriter(video_tmp_path, fourcc, fps, (w, h))
    for j in tqdm(range(num_images), position=0, desc='writing video'):
        img = cv2.imread(os.path.join(save_root, '%05d.png' % j))
        img = cv2.putText(
            img,
            text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            thickness,
            lineType,
        )
        out.write(img)

    out.release()

    print("ffmpeg version:")
    subprocess.call("ffmpeg -version", shell=True)

    # TODO: Set proper size of video: [-s {w}x{h}]
    cmd = f'ffmpeg -y -i "{video_tmp_path}" -i "{audio_path}" -acodec aac -vcodec h264 -shortest -threads 0 -s {w}x{h} "{output_path}"'
    # "-pix_fmt yuv420p -profile:v baseline -level 3"

    print(f"ffmpeg cmd: {cmd}")
    return_code = subprocess.call(cmd, shell=True)

    if return_code > 0:
        raise Exception(f"An error occurred when assembling the output video: ffmpeg return_code={return_code}")

    try:
        os.remove(video_tmp_path)  # remove the template video

    except FileNotFoundError:
        return


if __name__ == '__main__':

    opt = PostprocessingOptions().parse()

    DP = opt.dataset_target
    NAME = opt.model_name
    FILE_ID_SOURCE = opt.file_id_source
    FILE_ID_TARGET = opt.file_id_target
    frames_path = opt.frames_path
    audio_fname = opt.audio_fname
    target_fps = opt.target_fps
    results_out_dir = opt.results_out_dir

    h5py_path = os.path.join(DP, FILE_ID_TARGET, FILE_ID_TARGET + '.h5')

    cropped = h5py.File(h5py_path, 'r')["frame"]
    crop_image_shape = cropped[0].shape[:-1]

    images_out_path = os.path.join(DP, 'images')  # results_out_dir
    tform_path = os.path.join(DP, FILE_ID_TARGET, 'tform.npy')

    tform = np.load(tform_path)

    final_file = os.path.join(results_out_dir, f'{FILE_ID_SOURCE}_to_{FILE_ID_TARGET}.mp4')
    frames_out_path = os.path.join(DP, 'generated_frames')  # results_out_dir
    os.makedirs(frames_out_path, exist_ok=True)

    num_images = int(len(os.listdir(images_out_path)))

    for index, frame in zip(tqdm(range(num_images)), sorted(os.listdir(frames_path))):
        image_path = '%05d.png' % index

        image = cv2.imread(os.path.join(images_out_path, image_path))
        # resize image to cropped image size
        image = cv2.resize(image, crop_image_shape)

        old_image = cv2.imread(os.path.join(frames_path, frame))

        f_tform = SimilarityTransform(matrix=tform[index])

        res = warp_back(image, old_image, f_tform)

        cv2.imwrite(os.path.join(frames_out_path, image_path), res)

    h, w, _ = res.shape

    write_video_with_audio(frames_out_path, audio_fname, final_file, h=h, w=w, fps=target_fps)

    try:
        if opt.clean:
            shutil.rmtree(images_out_path)
            shutil.rmtree(frames_out_path)
            shutil.rmtree(frames_path)
            
    except Exception as e:
        print(e)
