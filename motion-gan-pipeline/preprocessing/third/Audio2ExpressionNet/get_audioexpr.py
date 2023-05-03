import sys
import os
import random
import torchvision.transforms as transforms
import argparse


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from data.audio_dataset import AudioDataset
from data.base_dataset import BaseDataset

from models import create_model
import torch
import numpy as np
import copy
from tqdm import tqdm

class FaceDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt

        # directories
        self.dataroot = opt.dataroot
        self.expr_path = os.path.join(opt.dataroot, 'deca_expr')
        self.audio_feature_path = os.path.join(opt.dataroot, 'audio_feature')

        # debug print
        print('load sequence:', self.dataroot)

        self.n_frames_total = len(os.listdir(self.expr_path)) - 1

        opt.nTrainObjects = 1
        opt.nValObjects = 1
        opt.nTestObjects = 1

        opt.test_sequence_names = [[opt.dataroot.split("/")[-1], 'train']]
        assert(opt.resize_or_crop == 'resize_and_crop')

    def getSampleWeights(self):
        weights = np.ones((self.n_frames_total))
        return weights


    def getAudioFilename(self):
        return os.path.join(self.dataroot, 'audio.wav')

    def getAudioFeatureFilename(self, idx):
        return os.path.join(self.audio_feature_path, str(idx) + '.deepspeech.npy')


    def __getitem__(self, global_index):
        # select frame from sequence
        index = global_index

        # expressions
        expressions = np.load(os.path.join(self.expr_path, '%05d.npy' % index))
        expressions = torch.from_numpy(expressions)[0]

        # identity
        identity = torch.zeros(100) # not used

        # load deepspeech feature
        feature_array = np.load(os.path.join(self.audio_feature_path, '%05d.deepspeech.npy' % index))
        dsf_np = np.resize(feature_array,  (16,29,1))
        dsf = transforms.ToTensor()(dsf_np.astype(np.float32))


        # load sequence data if necessary
        if self.opt.look_ahead:# use prev and following frame infos
            r = self.opt.seq_len//2
            for i in range(1,r): # prev frames
                index_seq = index - i
                if index_seq < 0: index_seq = 0

                feature_array = np.load(os.path.join(self.audio_feature_path, '%05d.deepspeech.npy' % index_seq))
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf_seq, dsf], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current]
            
            for i in range(1,self.opt.seq_len - r + 1): # following frames
                index_seq = index + i
                max_idx = self.n_frames_total-1
                if index_seq > max_idx: index_seq = max_idx

                feature_array = np.load(os.path.join(self.audio_feature_path, '%05d.deepspeech.npy' % index_seq))
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf, dsf_seq], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current ... future]
        else:
            last_valid_idx = audio_id
            for i in range(1,self.opt.seq_len):
                index_seq = index - i
                if index_seq < 0: index_seq = 0

                feature_array = np.load(os.path.join(self.audio_feature_path, '%05d.deepspeech.npy' % index_seq))
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf_seq, dsf], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current]


        #################################
        weight = 1.0 / self.n_frames_total
        
        return {#'TARGET': TARGET, 'UV': UV,
                'paths': '', #img_path,
                'expressions': expressions,
                'identity': identity,
                'audio_deepspeech': dsf, # deepspeech feature
                'target_id': -1,
                'internal_id': 0,
                'weight': np.array([weight]).astype(np.float32)}

    def __len__(self):
        return self.n_frames_total

    def name(self):
        return 'FaceDataset'


        
        self.expressions = input['expressions'].cuda()
        self.audio_features = input['audio_deepspeech'].cuda() # b x seq_len x 16 x 29
        self.target_id = input['target_id'].cuda()


def make_opts(dataset_base):
    class Struct:
        def __init__(self, **entries):
            self.__dict__.update(entries)

    opt = {}
    opt['dataroot'] = dataset_base
    opt['batch_size'] = 1
    opt['seq_len'] = 8
    opt['fineSize'] = 512
    opt['display_winsize'] = 512
    opt['input_nc'] = 3
    opt['output_nc'] = 3
    opt['ngf'] = 64
    opt['ndf'] = 64
    opt['netD'] = 'basic'
    opt['netG'] = 'unet_256'
    opt['n_layers_D'] = 3
    opt['gpu_ids'] = '0'
    opt['fineSize'] = 256

    opt['name'] = 'audio2ExpressionsAttentionTMP4-estimatorAttention-SL8-BS16-ARD_ZDF-multi_face_audio_eq_tmp_cached-RMS-20191105-115332-look_ahead'
    opt['renderer'] = 'no_renderer'
    opt['fix_renderer'] = True
    opt['dataset_mode']='multi_face_audio_eq_tmp_cached'
    opt['model'] = 'audio2ExpressionsAttentionTMP4'
    opt['direction'] = 'AtoB'
    opt['epoch'] = 'latest'
    opt['load_iter'] = 0
    opt['num_threads'] = 4
    opt['checkpoints_dir'] = os.getcwd() + '/third/Audio2ExpressionNet/checkpoints'
    opt['norm'] = 'instance'
    opt['serial_batches'] = False 
    opt['no_dropout'] = False
    opt['max_dataset_size'] = float("inf")

    opt['resize_or_crop'] = 'resize_and_crop'
    opt['no_augmentation'] = False
    opt['init_type'] = 'xavier'
    opt['init_gain'] = 0.02

    opt['verbose'] = False
    opt['suffix'] = ''
    opt['tex_dim'] = 256
    opt['tex_features_intermediate'] = 16
    opt['tex_features'] = 16
    opt['textureModel'] = 'DynamicNeuralTextureAudio'
    opt['rendererType'] = 'estimatorAttention'
    opt['lossType'] = 'RMS'

    opt['hierarchicalTex'] = False
    opt['output_audio_expressions'] = False
    opt['erosionFactor'] = 1.0
    opt['audio_window_size'] = 16
    opt['look_ahead'] = True
    opt['cached_images'] = False
    opt['ntest'] = float("inf")
    opt['results_dir'] = './results/'
    opt['aspect_ratio'] = 1.0,
    opt['phase'] = 'test'
    opt['eval'] = False
    opt['num_test'] = 50
    opt['write_no_images'] = True

    # Important
    opt['source_dir'] = './datasets/'
    opt['source_actor'] = dataset_base

    # extra
    opt['isTrain'] = False

    s = Struct(**opt)

    return s


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


def load_target_sequence(opt):
    opt_target = copy.copy(opt) # create a clone
    # opt_target.dataroot = opt.target_actor # overwrite root directory
    opt_target.dataset_mode = 'face'
    opt_target.phase = 'train'
    dataset_target = FaceDataset()
    dataset_target.initialize(opt_target)

    dataloader = torch.utils.data.DataLoader(
            dataset_target,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    return dataset_target, dataloader


def get_audioexpr(name, dataset_base, out_dir, mapping_path=None):

    # read options
    print('Base: ', dataset_base)
    opt = make_opts(dataset_base)

    # hard-code some parameters for test
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # no shuffle
    opt.no_augmentation = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.source_actor = dataset_base

    # load model
    model = load_model(opt)

    # Make mapping
    if mapping_path is None:
        mapping_fn = os.path.join(dataset_base, 'mapping.npy')
    
    else:
        mapping_fn = mapping_path

    not_exists = not os.path.exists(mapping_fn)

    if not_exists:
        # read target sequence
        dataset_target, data_loader_target = load_target_sequence(opt)

        # collect data
        print('collect data')
        audio_expressions = None
        gt_expressions = None
        for i, data in tqdm(enumerate(data_loader_target)):
            model.set_input(data)
            model.test()
            
            ae = model.fake_expressions.data[:,:,0]
            if type(audio_expressions) == type(None):
                audio_expressions = ae
                e = model.expressions.data
                gt_expressions = e
            else:
                audio_expressions = torch.cat([audio_expressions,ae],dim=0)
                e = model.expressions.data
                gt_expressions = torch.cat([gt_expressions,e],dim=0)

        # solve for mapping
        print('solve for mapping')
        optimize_in_parameter_space = True #False
        if optimize_in_parameter_space:
#            A = audio_expressions
#            B = gt_expressions
#            # solve lstsq  ||AX - B||
#            X, _ = torch.gels(B, A, out=None)
#            #X, _ = torch.lstsq(B, A) # requires pytorch 1.2
#            X = X[0:A.shape[1],:]
#            mapping = X.t()

            # use gradient descent method
            n = audio_expressions.shape[0]
            subspace_dim = 32

            # TODO: patch
            n_expr = 53

            X = torch.nn.Parameter(torch.randn(n_expr, subspace_dim, requires_grad=True).cuda())
            optimizer = torch.optim.Adam([X], lr=0.01)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
            num_epochs = 90
            random_range = [k for k in range(0,n)]
           
            for ep in tqdm(range(0,num_epochs)):
                random.shuffle(random_range)
                for j in random_range:
                    expressions = gt_expressions[j]
                    fake_expressions = torch.matmul(X, audio_expressions[j])
                    diff_expression = fake_expressions - expressions
                    loss = torch.mean(diff_expression * diff_expression) # L2
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                lr_scheduler.step()

            mapping = X.data
        
        map_cpu = mapping.data.cpu().numpy()
        np.save(mapping_fn, map_cpu)

    else:
        # load mapping from file
        map_cpu = np.load(mapping_fn)
        mapping = torch.tensor(map_cpu.astype(np.float32)).cuda()
        print('loaded mapping from file', mapping.shape)


    # read source sequence
    dataset_source, data_loader_source = load_source_sequence(opt)

    expression_multiplier = 1.0

    print(f'Extracting audio-features of {name}..')
    for i, data in enumerate(tqdm(data_loader_source)):
        model.set_input(data)
        model.test()
        audio_expression = model.fake_expressions.data[0, :, 0]
        expression = expression_multiplier * torch.matmul(mapping, audio_expression)
        expression = expression[None, :]
        np.save(os.path.join(out_dir, '%05d.npy' % i), expression.cpu().numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', required=True)
    parser.add_argument('--dataset_base', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--mapping_path', required=True)
    inopt = parser.parse_args()

    out_dir = os.path.join(inopt.out_dir, 'audio_expr')
    os.makedirs(out_dir, exist_ok=True)

    get_audioexpr(inopt.name, inopt.dataset_base, out_dir, inopt.mapping_path)
