# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
import shutil

def remove_folders(folders):
    print('\n\nCleaning folders..\n\n')
    for folder in folders:
        try:
            shutil.rmtree(folder)

        except:
            print(folder)
            pass
    return

def create_h5py_dataset(target_dir, clean=False):

    ds_path = os.path.join(target_dir, 'audio_feature')
    expr_path = os.path.join(target_dir, 'expressions')
    frames_path = os.path.join(target_dir, 'frames')
    uv_path = os.path.join(target_dir, 'uv')
    mask_path = os.path.join(target_dir, 'mask_mouth')
    deca_details_path = os.path.join(target_dir, 'deca_details')

    h5py_out_file = os.path.join(target_dir, f'{target_dir.split("/")[-1]}.h5')
    if os.path.isfile(h5py_out_file):
        print('H5py file already exists.')
        if clean:
            folders = [ds_path, expr_path, frames_path, uv_path, mask_path, deca_details_path]
            remove_folders(folders)
        return

    f = h5py.File(h5py_out_file, 'w')

    dataset_size = len(os.listdir(ds_path)) - 1
    print(f'Dataset size: {dataset_size}')
    
    # Load deepspeech data
    dsf_data = []
    for i in tqdm(range(dataset_size)):
        dsf_data.append(np.load(os.path.join(ds_path, f'{i}.deepspeech.npy')))

    dsf_data = np.array(dsf_data)
    grp = f.create_dataset("dsf", dsf_data.shape, data=dsf_data)

    # Load expression data
    try:
        ep_data = []
        for i in tqdm(range(dataset_size)):
            ep_data.append(np.load(os.path.join(expr_path, f'expr_{i}.npy')))

        ep_data = np.array(ep_data)
        grp = f.create_dataset("ep", ep_data.shape, data=ep_data)

    except FileNotFoundError:
        pass

    try:
        frame_data = []
        grp = f.create_dataset("frame", (dataset_size, 224, 224, 3))
        for i in tqdm(range(dataset_size)):
            im = np.asarray(Image.open(os.path.join(frames_path, '%04d.jpg' % i)))
            f["frame"][i] = im

    except FileNotFoundError:
        pass

    try:
        uv_data = []
        grp = f.create_dataset("uv", (dataset_size, 224, 224, 2))
        for i in tqdm(range(dataset_size)):
            im = np.asarray(np.load(os.path.join(uv_path, f'uv_{i}.npy')))
            f["uv"][i] = im

    except FileNotFoundError:
        pass

    try:
        mask_data = []
        grp = f.create_dataset("mask", (dataset_size, 224, 224))
        for i in tqdm(range(dataset_size)):
            im = np.asarray(np.load(os.path.join(mask_path, f'mask_{i}.npy')))
            f["mask"][i] = im

    except FileNotFoundError:
        pass

    try:
        deca_details_data = []
        grp = f.create_dataset("deca_details", (dataset_size, 224, 224, 3))
        for i in tqdm(range(dataset_size)):
            im = np.asarray(np.load(os.path.join(deca_details_path, f'deca_details_{i}.npy')))
            f["deca_details"][i] = im

    except FileNotFoundError:
        pass

    f.close()

    print('Created h5 file at : ', h5py_out_file)

    if clean:
        folders = [ds_path, expr_path, frames_path, uv_path, mask_path, deca_details_path]
        remove_folders(folders)



if __name__ == '__main__':

    dataset_type = 'SRF_anchor_short'
    dataset_name = 'Halbtotale_355_9414'
    dataset_path = '/home/alberto/NeuralVoicePuppetry/datasets'

    create_h5py_dataset(dataset_type, dataset_name, dataset_path)
