# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from email.policy import default
from .base_options_audio2headpose import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)

        # logging options
        parser.add_argument('--tf_log', type=bool, default=True, help='Use tensorboard logs')
        parser.add_argument('--display_winsize', type=int, default=256, help='display window size for both visdom and HTML')
        
        # visdom and HTML visualization parameters
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default='10000', help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving intermediate checkpoints during training')
        parser.add_argument('--save_epoch_freq', type=int, default=25, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', default=False, action='store_true', help='continue training: load the latest model')
        parser.add_argument('--load_epoch', type=str, default='0', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--re_transform', type=int, default=0, help='re-transform landmarks')
        
        
        # training parameters
        parser.add_argument('--smooth_loss', type=int, default=0, help='use smooth loss weight, 0 for not use')
        # parser.add_argument('--train_dataset_names', type=str, default='train_list.txt', help='chooses validation datasets.')
        # parser.add_argument('--validate_dataset_names', type=str, default='val_list.txt', help='chooses validation datasets.')
        parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs')
        parser.add_argument('--lr_policy', type=str, default='step', help='learning rate policy. [linear | step | plateau | cosine]')    
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--gamma', type=float, default=0.2, help='step learning rate gamma')
        parser.add_argument('--lr_decay_iters', type=int, default=250, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--n_epochs_decay', type=int, default=250, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--validate_epoch', type=int, default=50, help='validate model every some epochs, 0 for not validate during training')
        parser.add_argument('--loss_smooth_weight', type=float, default=0, help='smooth loss weight, 0 for not use smooth loss')
        parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam, AdamW, RMSprop')
        
        
        # data augmentations
        parser.add_argument('--gaussian_noise', type=int, default=1, help='whether add gaussian noise to input & groundtruth features')
        parser.add_argument('--gaussian_noise_scale', type=float, default=0.01, help='gaussian noise scale')
               

        self.isTrain = True
        return parser
