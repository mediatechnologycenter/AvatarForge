# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import numpy as np
from skimage.color import rgb2gray
from skimage.registration import optical_flow_tvl1
from PIL import Image
from tqdm import tqdm
import os

def compute_optical_flow_slow(frames, opticalflow_dir, debug_opticalflow_dir):

    for i in tqdm(range(len(frames)-1)):
        out_name = os.path.join(opticalflow_dir,'%05d.npy' % i)
        # --- Load the sequence
        image0 = np.array(Image.open(frames[i]))
        image1 = np.array(Image.open(frames[i+1]))

        # --- Convert the images to gray level: color is not supported.
        image0 = rgb2gray(image0)
        image1 = rgb2gray(image1)

        # --- Compute the optical flow
        v, u = optical_flow_tvl1(image0, image1)

        of = np.stack((v,u),axis=0)
        np.save(out_name, of)

    return

def compute_optical_flow(frames, opticalflow_dir, debug_opticalflow_dir):
    from mmflow.apis import inference_model, init_model
    from mmflow.datasets import visualize_flow, write_flow

    import mmcv

    # Specify the path to model config and checkpoint file
    config_file = 'third/mmflow/flownet2CS/flownet2cs_8x1_sfine_flyingthings3d_subset_384x768.py'
    checkpoint_file = 'third/mmflow/flownet2CS/flownet2cs_8x1_sfine_flyingthings3d_subset_384x768.pth'
    device = 'cuda:0'

    # init a model
    model = init_model(config_file, checkpoint_file, device=device)

    # inference the demo image
    for i in tqdm(range(len(frames)-1)):
        out_name = os.path.join(opticalflow_dir,'%05d.flo' % i) 
        result = inference_model(model, frames[i] , frames[i+1])
        write_flow(result, flow_file=out_name)
        # save the visualized flow map
        debug_file = os.path.join(debug_opticalflow_dir, '%05d.png' % i)
        flow_map = visualize_flow(result, save_file=debug_file)
