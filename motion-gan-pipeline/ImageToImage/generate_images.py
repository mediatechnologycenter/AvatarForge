# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from email.policy import default
from PIL import Image
import torch
import PIL
import configargparse
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np

from models.unet import UNet
from datasets.base import build_dataloader
from utils.utils import create_image_pair, save_image_list

import warnings
warnings.filterwarnings("ignore")

def config_parser():
    parser = configargparse.ArgumentParser()
    # config file
    parser.add_argument('-c', '--my-config', is_config_file=True, help='config file path', default='config/test.yml')
    # dataloader options
    parser.add_argument("--mode", type=str, default='test', help="test mode has if no ground truth data available, val otherwise")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers to use during batch generation")
    parser.add_argument("--num_input_channels", type=int, default=3, help="number of input image channels")
    parser.add_argument("--num_output_channels", type=int, default=3, help="number of output image channels")
    parser.add_argument("--use_label_maps", action='store_true', help="choose if to use label maps for discriminator")

    # dataset options
    parser.add_argument("--dataset_type", type=str, help="options: CustomDataset", default='CustomDataset')  
    parser.add_argument("--input_test_root_dir", type=str, help="Path to test input images", default='./data/input_images_test')
    parser.add_argument("--input_val_root_dir", type=str, help="Path to val input images", default='./data/input_images_val')  
    parser.add_argument("--label_val_root_dir", type=str, help="Path to val label images", default='./data/label_images_val')   
    parser.add_argument("--output_val_root_dir", type=str, help="Path to val output images", default='./data/output_images_val')  
    parser.add_argument("--width", type=int, default=640, help="width")
    parser.add_argument("--height", type=int, default=360, help="height")
    parser.add_argument("--metric_names", nargs="*", type=str, default=['mean_absolute_error'], help="names of metrics to be logged")
    parser.add_argument("--metric_weights", nargs="*", type=float, default=[1.0], help="weights assigned to metrics in the order")
    # logging/saving options
    parser.add_argument("--video_name", type=str, help='name of the reference video', default='Clara')
    parser.add_argument("--out_dir", type=str, help='directory in which to save result images', default='./results/')
    # dataroot
    parser.add_argument("--dataroot", required=True, type=str, help='input data dataroot', default='')
    parser.add_argument("--checkpoint_dir", required=True, type=str, help='path to checkpoint folder', default='../checkpoints/')
    return parser


def load_model(network, args, device):
    if Path(args.load_path).exists():
        checkpoint = torch.load(args.load_path, map_location=device)
        network.load_state_dict(checkpoint['model_state_dict'])
        try:
            args.continue_from_epoch = checkpoint['epoch']+1
            print("-> loaded model %s (epoch: %d)"%(args.load_path, args.continue_from_epoch))

        except TypeError:
            args.continue_from_epoch = None
            print("-> loaded model %s (epoch: final)"%(args.load_path))


if __name__=='__main__':

    parser = config_parser()
    args = parser.parse_args()

    # Overwrite image dimentions
    edge_path = os.path.join(args.dataroot, args.video_name, 'edges')
    # img_size = Image.open(os.path.join(edge_path, os.listdir(edge_path)[0])).size
    try: 
        img_size = np.load(os.path.join(args.dataroot, args.video_name, 'img_size.npy'))
    except FileNotFoundError:
        img_size = Image.open(os.path.join(edge_path, os.listdir(edge_path)[0])).size
    
    args.width, args.height = img_size
    ratio =  args.height / args.width

    print('Img Ratio: ', ratio)

    # resize for training quickly 
    if args.width >= args.height:
        args.width = 720
        args.height = int(args.width * ratio)
    
    else:
        args.height = 720
        args.width = int(args.height / ratio)
    
    os.makedirs(os.path.join(args.checkpoint_dir, args.video_name), exist_ok=True)
    args.load_path = os.path.join(args.checkpoint_dir, args.video_name, 'latest_GAN_model.pt')

    os.makedirs(args.out_dir, exist_ok=True)

    # Normal training
    temporal = True
    optical = False

    if not os.path.isfile(args.load_path):
        print('Checkpoint not found!')
        chkp_dir = os.path.join(args.checkpoint_dir, args.video_name)
        data_input_path = os.path.join(args.dataroot, args.video_name)

        if temporal:
            os.system('python train_temporal.py -c config/train_temporal.yml --checkpoints_dir ' + chkp_dir +
                    ' --input_train_root_dir ' + data_input_path +
                    '/edges --output_train_root_dir ' + data_input_path +
                    '/cropped --height ' + str(args.height) + ' --width ' + str(args.width) + ' --skip_log')
            
        elif optical:
            os.system('python train_optical.py -c config/train_optical.yml --checkpoints_dir ' + chkp_dir +
                    ' --input_train_root_dir ' + data_input_path +
                    '/edges --output_train_root_dir ' + data_input_path +
                    '/cropped --flow_train_root_dir ' + data_input_path +
                    '/opticalflow --height ' + str(args.height) + ' --width ' + str(args.width))
        
        else:
            os.system('python train.py -c config/train.yml --checkpoints_dir ' + chkp_dir +
                    ' --input_train_root_dir ' + data_input_path +
                    '/edges --output_train_root_dir ' + data_input_path +
                    '/cropped' + ' --height ' + str(args.height) + ' --width ' + str(args.width))
        

    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    # initialize the dataloaders
    test_loader = build_dataloader(args, mode=args.mode, shuffle=False) 

    # build the network 
    network = UNet(args).to(device)

    load_model(network, args, device)
    network.eval()

    # run inference
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
            
            inputs = data['input_image'].to(device)
            names = data['name']

            prediction = network(inputs)

            images = create_image_pair([inputs, prediction])
            images_output = create_image_pair([prediction])
            
            # save_image_list(images, args.results_dir, names)
            save_image_list(images_output, args.out_dir, names)

