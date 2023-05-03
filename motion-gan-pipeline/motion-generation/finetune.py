# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import imp
import os
import time
import yaml
from options.train_audio2headpose_options import TrainOptions
from datasets import create_dataset
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
import argparse
from util.cfgnode import CfgNode
import numpy as np


def finetune(name, dataroot, dataset_names, target_checkpoints, checkpoint_path, dataset_mode='deepspeech', fps=25):
    print('Fine-tuning model..')
    # Load default options
    Train_parser = TrainOptions()
    opt = Train_parser.parse()   # get training options
    
    # Overwrite with input
    opt.name = name
    opt.dataset_mode = dataset_mode
    opt.dataroot = os.path.join(dataroot, 'video')
    opt.dataset_names = dataset_names
    opt.FPS = fps

    # Finetune params
    opt.n_epochs = 15
    opt.n_epochs_decay = 10
    opt.checkpoints_dir = target_checkpoints
    opt.train_dataset_names = [os.path.join(opt.dataroot, opt.dataset_names)]
    opt.validate_dataset_names = []
    opt.train_test_split = False
    opt.lr = 1e-4

    # save to the disk
    Train_parser.print_options(opt)

    # Load data
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    model.load_checkpoint(checkpoint_path) # Load checkpoint to finetune   
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    for epoch in tqdm(range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1)):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        # visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            # if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
            #     losses = model.get_current_losses()
            #     t_comp = (time.time() - iter_start_time) / opt.batch_size
            #     visualizer.print_current_errors(epoch, total_iters, losses, t_data)
            #     if opt.display_id > 0:
            #         visualizer.plot_current_errors(losses, total_iters)

            iter_data_time = time.time()
            
        if epoch % opt.save_epoch_freq == 0 and epoch!=0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('epoch_%d' % (epoch))
        
        # early stopping
        losses = model.get_current_losses()
        if losses['GMM'] <= 0:
            print('Negative loss at the end of epoch %d, total iters %d' % (epoch, total_iters))
            print('Stopping Training')
            break

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
    
    model.save_networks('latest')

if __name__ == '__main__':

    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='', help="person name, e.g. Obama1, Obama2, May, Nadella, McStay")    
    parser.add_argument('--dataset_mode', default='deepspeech', help="type of dataset")
    parser.add_argument('--dataroot', help="Path to data folders")
    parser.add_argument('--dataset_names', default='', help="Nme of the dataset")
    parser.add_argument('--fps', default=25, help="target fps")
    parser.add_argument('--target_checkpoints', default="", help="Path to the output checkpoint directory")
    parser.add_argument('--checkpoint_path', default="./checkpoints/latest_Audio2Headpose.pkl", help="Path to the checkpoints to finetune")

    inopt = parser.parse_args()

    finetune(inopt.name, inopt.dataroot, inopt.dataset_names, inopt.target_checkpoints, inopt.dataset_mode, inopt.fps, inopt.checkpoint_path)
