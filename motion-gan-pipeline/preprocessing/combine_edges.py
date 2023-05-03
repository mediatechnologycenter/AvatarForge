# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from ast import Num
from genericpath import isfile
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from scipy.spatial import KDTree
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from PIL import Image
import argparse
from edge_creation.utils import get_edge_predicted, get_edge_image_mixed, get_crop_coords, convert_to_rgb
from scipy.ndimage import gaussian_filter1d

def combine_edges(inopt):
    # load generated headposes
    generated_hp_path = os.path.join(inopt.out_dir, 'headposes.npy')
    generated_hp = np.load(generated_hp_path)

    # load tracked headposes from target video
    track_params_path = os.path.join(inopt.dataset_base, 'track_params.pt')
    tracked_params = torch.load(track_params_path)
    tracked_R = tracked_params['euler'].numpy()
    tracked_t = tracked_params['trans'].numpy()
    tracked_hp = np.concatenate((tracked_R, tracked_t), axis=1)

    # Compute crop size
    img = Image.open(os.path.join(inopt.dataset_base, 'matting', '%05d.png' % 0))
    img = convert_to_rgb(img)

    ldk_paths = os.listdir(os.path.join(inopt.out_dir, 'landmarks'))
    points = np.mean([np.loadtxt(os.path.join(inopt.out_dir, 'landmarks', ldk))[:,:2] for ldk in ldk_paths], axis=0)
    crop_coords = get_crop_coords(points, img.size)

    for frame_num in tqdm(range(generated_hp.shape[0])):

        ist, index = KDTree(tracked_hp).query(generated_hp[frame_num], workers=-1)

        img_path = os.path.join(inopt.dataset_base, 'matting', '%05d.png' % index)
        generated_landmark_path = os.path.join(inopt.out_dir, 'landmarks', '%05d.lms' % frame_num)
        tracked_landmark_path = os.path.join(inopt.dataset_base, 'landmarks', '%05d.lms' % index)

        get_edge_image_mixed(inopt.out_dir, img_path, generated_landmark_path, tracked_landmark_path, crop_coords, frame_num)
        

def predict_face2body(inopt):

    # check device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # load head2body model
    checkpoint_path = os.path.join(inopt.checkpoint_dir, inopt.target_name, 'head2body.pkl')
    if not os.path.isfile(checkpoint_path):
        print('No checkpoint found.')
        exit()

    model = model = nn.Linear(77 * 2, 4 * 2).to(device)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # landmarks
    num_ldk = len(os.listdir(os.path.join(inopt.out_dir, 'landmarks')))

    # transforms
    trans = transforms.ToTensor()

    # get img size
    tmp_img = Image.open(os.path.join(inopt.out_dir, 'render', '00000.png'))
    im_size = tmp_img.size
    w, h = im_size
    
    print('Predicting body landmarks..')
    output_ldks = []
    for idx in tqdm(range(num_ldk)):
        landmark_name = os.path.join(inopt.out_dir, 'landmarks', '%05d.lms' % idx)

        # load and transform input
        input_ldk = np.loadtxt(landmark_name)[:, :2]
        input_ldk[:, 0] = input_ldk[:, 0] / w
        input_ldk[:, 1] = input_ldk[:, 1] / h
        input_ldk = trans(input_ldk).flatten().float().to(device).unsqueeze(0)

        # generate and scale output
        output_ldk = model(input_ldk)

        # Transform 
        input_ldk = input_ldk.cpu().detach().reshape((77, 2)).numpy()
        input_ldk[:, 0] = input_ldk[:, 0] * w
        input_ldk[:, 1] = input_ldk[:, 1] * h

        output_ldk = output_ldk.cpu().detach().reshape((4, 2)).numpy()
        output_ldk[:, 0] = output_ldk[:, 0] * w
        output_ldk[:, 1] = output_ldk[:, 1] * h

        output_ldks.append(output_ldk)

    # Smooth predicted landmarks
    smooth_sigma = 5
    output_ldks = np.array(output_ldks)
    output_ldks = gaussian_filter1d(output_ldks.reshape(-1, 8), smooth_sigma, axis=0).reshape(-1, 4, 2)

    print('\nMaking edges..')
    for idx in tqdm(range(num_ldk)):
        landmark_name = os.path.join(inopt.out_dir, 'landmarks', '%05d.lms' % idx)
        output_ldk = output_ldks[idx]
        get_edge_predicted(idx, inopt.out_dir, landmark_name, output_ldk, im_size)

    

if __name__=='__main__':

    # load args 
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_base', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--target_name', required=True)
    parser.add_argument('--checkpoint_dir', required=True)

    inopt = parser.parse_args()

    # call
    # combine_edges(inopt)

    predict_face2body(inopt)

