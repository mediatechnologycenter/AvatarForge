import argparse


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


class PreprocessingOptions(Options):

    def initialize(self, parser):

        parser = Options.initialize(self, parser)
        parser.add_argument('--dataroot', required=True, default='/home/alberto/data/nerf-videosynth/',
                            help='path to data folder')

        parser.add_argument('--name', required=True, help='name of video/audio to process')

        parser.add_argument('--target_fps', type=int, default=25, help='Fps of target video')

        parser.add_argument('--preprocessing_type', type=str, default='video', help='Preprocessing for video or audio')

        parser.add_argument('--audioexpr', action='store_true', help='Extracts NVP audio-expressions')

        parser.add_argument('--step', type=str, default='0', help='Pre-processing step to take')

        parser.add_argument('--use_DECA', action='store_true', help='If true, use deca tracker for face model')
        parser.add_argument('--use_FLAME', action='store_true', help='If true, use flame tracker for face model')
        parser.add_argument('--use_BASEL', action='store_true', help='If true, use basel tracker for face model')

        parser.add_argument('--train_split', type=float, default=0.9, help='Percentage of data used for training')
        parser.add_argument('--val_split', type=float, default=0.01, help='Percentage of data used for validation')

        return parser

