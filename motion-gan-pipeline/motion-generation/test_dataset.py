# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import yaml
from options.train_audio2headpose_options import TrainOptions
from datasets import create_dataset
import argparse
from util.cfgnode import CfgNode


if __name__ == '__main__':
    # Load default options
    Train_parser = TrainOptions()
    opt = Train_parser.parse()   # get training options

    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Audio2Headpose_MTC.yml', help="person name, e.g. Obama1, Obama2, May, Nadella, McStay")    
    inopt = parser.parse_args()
    # TODO: make config
    conf_debug = 'config/Audio2Headpose_Ted.yml'
    # with open(inopt.config) as f:
    with open(conf_debug) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)
    
    # Overwrite with config
    opt.name = cfg.experiment.name
    opt.dataset_mode = cfg.experiment.dataset_mode
    opt.dataroot = cfg.experiment.dataroot
    opt.dataset_names = cfg.experiment.dataset_names

    # Load data
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training points = %d' % dataset_size) # 80650

    dataset = dataset.dataset

    for i in range(len(dataset)):
        A2Hsamples, history_info, target_info = dataset[i]
        # print(f'Item: {i}, audio: {A2Hsamples.size()}, history: {history_info.size()}, target: {target_info.size()}')
