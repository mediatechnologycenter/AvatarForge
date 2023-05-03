
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


if __name__ == '__main__':
    # Load default options
    Train_parser = TrainOptions()
    opt = Train_parser.parse()   # get training options

    # Load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/Audio2Headpose_Mine.yml', help="person name, e.g. Obama1, Obama2, May, Nadella, McStay")    
    inopt = parser.parse_args()
    # TODO: make config
    with open(inopt.config) as f:
        cfg_dict = yaml.load(f, Loader=yaml.FullLoader)
        cfg = CfgNode(cfg_dict)

    # Overwrite with config
    opt.name = cfg.experiment.name
    opt.dataset_mode = cfg.experiment.dataset_mode
    opt.dataroot = cfg.experiment.dataroot
    opt.dataset_names = cfg.experiment.dataset_names
    opt.FPS = cfg.experiment.fps

    # save to the disk
    Train_parser.print_options(opt)

    # Load data
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
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

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_errors(epoch, total_iters, losses, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(losses, total_iters)

            iter_data_time = time.time()
            
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
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