import os
import os.path
from options.transfer_options import Audio2ExprOptions
from data import CreateDataLoader
from data.face_dataset import FaceDataset
from data.audio_dataset import AudioDataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import time
import random
import progressbar
import copy
from shutil import copyfile

from BaselModel.basel_model import *


def load_model(opt):
    opt.output_audio_expressions = True
    opt.nTrainObjects = 116

    print('#train objects = %d' % opt.nTrainObjects)

    print('>>> create model <<<')
    model = create_model(opt)
    print('>>> setup model <<<')
    model.setup(opt)

    return model

def load_source_sequence(opt):
    opt_source = copy.copy(opt)  # create a clone
    opt_source.dataroot = opt.source_actor  # overwrite root directory
    print(opt_source.dataroot)
    opt_source.dataset_mode = 'audio'
    opt_source.phase = 'train'

    dataset_source = AudioDataset()

    dataset_source.initialize(opt_source)

    dataloader = torch.utils.data.DataLoader(
        dataset_source,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads))

    return dataset_source, dataloader


if __name__ == '__main__':
    # read options
    opt = Audio2ExprOptions().parse()

    # load model
    model = load_model(opt)
    print('model version:', opt.name)

    if opt.use_mapping:
        # read mapping
        mapping_fn = opt.mapping_path

        # load mapping from file
        map_cpu = np.load(mapping_fn + '.npy')
        mapping = torch.tensor(map_cpu.astype(np.float32)).cuda()
        print('loaded mapping from file', mapping.shape)

    # make outdir
    source_name = opt.source_actor.split("/")[-1]
    out_dir = opt.out_dir + source_name
    os.makedirs(out_dir, exist_ok=True)
    audio2expr_dir = os.path.join(out_dir, 'audio2expr')
    os.makedirs(audio2expr_dir, exist_ok=True)
    if opt.use_mapping:
        expr_dir = os.path.join(out_dir, 'expr')
        os.makedirs(expr_dir, exist_ok=True)

    # read source sequence
    dataset_source, data_loader_source = load_source_sequence(opt)
    dataset_source_size = len(dataset_source)
    print('#source_actor  frames = %d' % dataset_source_size)

    expression_multiplier = 1.0  # default

    # run over data
    with progressbar.ProgressBar(max_value=len(dataset_source)) as bar:
        for i, data in enumerate(data_loader_source):
            bar.update(i)
            model.set_input(data)
            model.test()
            audio_expression = model.fake_expressions.data[0, :, 0]

            if opt.use_mapping:
                expression = expression_multiplier * 10.0 * torch.matmul(mapping, audio_expression)
                expression = expression[None, :]
                np.save(os.path.join(expr_dir, f'expr_{i}.npy'), expression.cpu().numpy())

            audio_expression = audio_expression[None, :]
            np.save(os.path.join(audio2expr_dir, f'audioexpr_{i}.npy'), audio_expression.cpu().numpy())

    exit()