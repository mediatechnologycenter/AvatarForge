# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import argparse
import os

class Options():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)


    def parse(self):

        opt = self.gather_options()

        self.print_options(opt)

        self.opt = opt
        return self.opt

class PostprocessingOptions(Options):

    def initialize(self, parser):

        parser = Options.initialize(self, parser)
        parser.add_argument('--file_id_source', required=True, help='Name of source')
        parser.add_argument('--file_id_target', required=True, help='Name of target')
        parser.add_argument('--model_name', required=True, help='Name of the rendering model')

        parser.add_argument('--frames_path', required=True, help='path to target orignal frames')
        parser.add_argument('--audio_fname', required=True, help='path to input audio file')
        parser.add_argument('--dataset_target', required=True, help='path to target dataset folder')
        parser.add_argument('--target_fps', type=int, default=25, help='Fps of target video')
        parser.add_argument('--clean', action='store_true', help='True to clean all data and inference folders')
        parser.add_argument('--results_out_dir', required=True, help='path to output folder')

        return parser


class PreprocessingOptions(Options):

    def initialize(self, parser):

        parser = Options.initialize(self, parser)
        parser.add_argument('--dataroot', required=True, default='/home/alberto/data/videosynth/',
                            help='path to data folder')
        parser.add_argument('--dataset_path', required=True, default='/home/alberto/NeuralVoicePuppetry/datasets/',
                            help='path to the dataset folder')
        parser.add_argument('--dataset', required=True, help='name of the dataset cathegory')
        parser.add_argument('--name', required=True, help='name of video/audio to process')

        parser.add_argument('--preprocess_ds', action='store_true', help='True to run deepspeach preprocessing')
        parser.add_argument('--preprocess_tracking', action='store_true', help='True to run tracking preprocessing')
        parser.add_argument('--skip_h5py', action='store_true', help='True to skip h5py creation')
        parser.add_argument('--target_fps', type=int, default=25, help='Fps of target video')
        parser.add_argument('--clean', action='store_true', help='True to clean all datset folders and leave only h5py')

        return parser

class FaceReconstructionOptions(Options):

    def initialize(self, parser):
        parser = Options.initialize(self, parser)
        parser.add_argument('--source_name', required=True, help='name of source dataset')
        parser.add_argument('--target_name', required=True, help='name of target dataset')
        parser.add_argument('--audio_path', required=True, help='path to source audio file')
        parser.add_argument('--expression_path', required=True, help='path to expressions dir')
        parser.add_argument('--codedicts_path', required=True, help='path to codedicts dir')
        parser.add_argument('--tform_path', required=True, help='path to tform file')
        parser.add_argument('--frames_path', required=True, help='path to frames dir')

        return parser

class Text2SpeachOptions(PreprocessingOptions):
    def initialize(self, parser):

        parser = PreprocessingOptions.initialize(self, parser)
        parser.add_argument('--language_source', type=str, default='en', help='Original language of text file')
        parser.add_argument('--language_target', type=str, default='en', help='Target language for audio file')

        return parser
