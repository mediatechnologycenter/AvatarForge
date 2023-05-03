# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
import os
import numpy as np
import h5py
import torch
from PIL import Image
from tqdm import tqdm

from autils.make_h5py import create_h5py_dataset
from base_options import PreprocessingOptions


def find_file_ext(data_path, dataset_name, name):
    dataset = dataset_name + '/' + name
    tmp = os.path.join(data_path, dataset, name + '.mp4')

    if os.path.isfile(tmp):
        return name + '.mp4'

    else:
        tmp = os.path.join(data_path, dataset, name + '.mp3')

        if os.path.isfile(tmp):
            return name + '.mp3'

        else:
            tmp = os.path.join(data_path, dataset, name + '.wav')
            if os.path.isfile(tmp):
                return name + '.wav'

            else:
                tmp = os.path.join(data_path, dataset, name + '.avi')

                if os.path.isfile(tmp):
                    return name + '.avi'

                else:
                    print('No input file with given name')
                    exit()


def deepspeech_preprocessing(name, target_dir, folder_videos, video_name, target_base, type, target_fps):
    from autils.deepspeech_features import extract_ds
    print(f'\n\nExtracting DeepSpeech features for {name}..\n\n')

    if not os.path.isfile(os.path.join(target_dir, target_dir.split("/")[-1] + '.h5')):
        extract_ds(folder_videos=folder_videos,
                   file_id=video_name.split("/")[-1][:-4],
                   target_base=target_base,
                   target_name=name,
                   type=type,
                   target_fps=target_fps)

    else:
        print('Already done.')


def video_tracking(tracker, target_dir):

    exp_dir = os.path.join(target_dir, 'expressions')
    os.makedirs(exp_dir, exist_ok=True)

    h5_path = os.path.join(target_dir, target_dir.split("/")[-1] + '.h5')
    if os.path.isfile(h5_path):
        data = h5py.File(h5_path, 'r')
        if "ep" in data.keys():
            print('Already done.')
            return

    codedict_dir = os.path.join(target_dir, 'DECA_codedicts')
    os.makedirs(codedict_dir, exist_ok=True)

    uv_dir = os.path.join(target_dir, 'uv')
    os.makedirs(uv_dir, exist_ok=True)

    frames_dir = os.path.join(target_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    deca_details_dir = os.path.join(target_dir, 'deca_details')
    os.makedirs(deca_details_dir, exist_ok=True)

    mask_dir = os.path.join(target_dir, 'mask_mouth')
    os.makedirs(mask_dir, exist_ok=True)

    tforms_file = os.path.join(target_dir, 'tform.npy')

    expressions = []
    tforms = []

    # Load video
    for i in tqdm(range(len(tracker.testdata))):
        images = tracker.testdata[i]['image'].to(tracker.device)[None, ...]
        tform = tracker.testdata[i]['tform']
        tform = np.array(tform.params)
        # og_image = tracker.testdata[i]['original_image'].cpu()
        name = tracker.testdata[i]['imagename']

        # save image and tfrom
        to_save = images[0].permute(1, 2, 0).cpu().numpy()
        to_save = to_save * 255.
        to_save = to_save.astype(np.uint8)

        filename = os.path.join(frames_dir, '%04d.jpg' % i)
        img = Image.fromarray(to_save)
        img.save(filename)

        tforms.append(tform)

        # call deca
        codedict, opdict, mask = tracker(images)

        # Save codedict
        pt_file = os.path.join(codedict_dir, f'codedict_{i}.pt')
        torch.save(codedict, pt_file)

        # Save expressions
        expression_params = codedict['exp'].cpu().numpy()[0]
        pose_params = codedict['pose'].cpu().numpy()[0, 3:]
        new_expr = np.concatenate((expression_params, pose_params), axis=0)

        exp_file = os.path.join(exp_dir, f'expr_{i}.npy')
        np.save(exp_file, new_expr)

        # Save uv
        uv_img = opdict['grid'].cpu()[0]
        npy_file = os.path.join(uv_dir, f'uv_{i}.npy')
        np.save(npy_file, uv_img)

        # Save mask
        mask_img = mask.cpu()
        npy_file = os.path.join(mask_dir, f'mask_{i}.npy')
        np.save(npy_file, mask_img)

        # Save deca details
        deca_details_img = opdict['detail_normal_images'][0].permute(1, 2, 0).cpu().numpy()
        npy_file = os.path.join(deca_details_dir, f'deca_details_{i}.npy')
        np.save(npy_file, deca_details_img)

    tforms = np.array(tforms)
    print(tforms.shape)
    np.save(tforms_file, tforms)


def preprocess_video(name, folder_videos, target_base, video_name, preprocess_ds, preprocess_tracking, skip_h5py, type, target_fps, clean):
    print('Video name: ', video_name)
    target_dir = os.path.join(target_base, name)
    print('target_dir: ', target_dir)
    target_file_name = video_name.split("/")[-1][:-4]
    print('target_file_name: ', target_file_name)
    print('folder_videos: ', folder_videos)

    frames_folder = target_base + '/' + target_file_name + '/og_frames'
    print('Video folder: ', frames_folder)

    if preprocess_ds:
        deepspeech_preprocessing(name, target_dir, folder_videos, video_name, target_base, type, target_fps)

    if preprocess_tracking:
        from autils.deca_flame_fitting import DECA_tracker

        print(f'\n\nExtracting Tracking information for {name}..\n\n')
        if os.path.isdir(frames_folder):

            tracker = DECA_tracker(frames_folder)

        else:
            tracker = DECA_tracker(folder_videos + '/' + target_file_name + '.mp4', target_dir=frames_folder)

        video_tracking(tracker, target_dir)

    if skip_h5py:
        return

    else:
        create_h5py_dataset(target_dir, clean)


if __name__ == '__main__':

    opt = PreprocessingOptions().parse()

    data_path = opt.dataroot
    dataset_path = opt.dataset_path
    dataset = opt.dataset
    name = opt.name
    target_fps = opt.target_fps
    skip_h5py = opt.skip_h5py
    clean = opt.clean

    full_path = find_file_ext(data_path, dataset, name)

    target_base = dataset_path

    # Flags to control preprocessing
    preprocess_ds = opt.preprocess_ds
    preprocess_tracking = opt.preprocess_tracking

    file_name = f'{full_path.split("/")[-1][:-4]}'
    print(file_name)
    folder_videos = os.path.join(data_path, dataset, file_name)

    if full_path.endswith("mp3") or full_path.endswith("wav"):
        preprocess_video(file_name, folder_videos, target_base, full_path,
                         preprocess_ds, False, skip_h5py, 'audio', target_fps, clean)

    else:
        preprocess_video(file_name, folder_videos, target_base, full_path,
                         preprocess_ds, preprocess_tracking, skip_h5py, 'video', target_fps, clean)
