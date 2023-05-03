import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from data.base_dataset import BaseDataset
from data.audio import Audio
from PIL import Image
from util import util
import h5py


def make_dataset(dir):
    images = []
    ids = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['.npy', '.NPY']):
                #.deepspeech.npy
                id_str = fname[:-15] #4]
                i = int(id_str)
                ids.append(i)
    ids = sorted(ids)

    for id in ids:
        fname=str(id)+'.deepspeech.npy'
        path = os.path.join(root, fname)
        images.append(path)
    return images

def make_expr_dataset(dir):
    expressions = []
    num_expr = len([name for name in os.listdir(dir) if os.path.isfile(name)])
    for i in range(num_expr):
        fname = f'expr_{i}.npy'
        path = os.path.join(dir, fname)
        expressions.append(path)
    return expressions

def make_ids(paths, root_dir):
    ids = []
    
    for fname in paths:
        l = fname.rfind('/')
        id_str = fname[l+1:-15]#4]
        i = int(id_str)
        #print(fname, ': ', i)
        ids.append(i)
    return ids


# def load_expressions(input_dir):
#     file = open(input_dir+"/expressions/", "r")
#     expressions = [[float(x) for x in line.split()] for line in file]
#     file.close()
#     return expressions

class FaceDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt

        # directories
        self.dataroot = opt.dataroot
        self.h5py_path = os.path.join(opt.dataroot, opt.dataroot.split("/")[-1]+'.h5')

        # debug print
        print('load sequence:', self.dataroot)
        print('\th5py_path:', self.h5py_path)

        self.data = h5py.File(self.h5py_path, 'r')

        # set data
        self.n_frames_total = min(len(self.data["dsf"]), len(self.data["ep"]))

        print('\tnum frames:', self.n_frames_total)


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
        #return self.frame_paths[idx % len(self.frame_paths)]
        audio_id = self.audio_ids[idx]
        return os.path.join(self.audio_feature_dir, str(audio_id) + '.deepspeech.npy')


    def __getitem__(self, global_index):
        # select frame from sequence
        index = global_index

        # intrinsics and extrinsics
        intrinsics = np.zeros((4))  # not used
        extrinsics = np.zeros((4,4))# not used

        # expressions
        expressions = self.data["ep"][index]
        expressions = torch.tensor(expressions)

        # identity
        identity = torch.zeros(100) # not used

        # load deepspeech feature
        feature_array = self.data["dsf"][index]
        dsf_np = np.resize(feature_array,  (16,29,1))
        dsf = transforms.ToTensor()(dsf_np.astype(np.float32))


        # load sequence data if necessary
        if self.opt.look_ahead:# use prev and following frame infos
            r = self.opt.seq_len//2
            for i in range(1,r): # prev frames
                index_seq = index - i
                if index_seq < 0: index_seq = 0

                feature_array = self.data["dsf"][index_seq]
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf_seq, dsf], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current]
            
            for i in range(1,self.opt.seq_len - r + 1): # following frames
                index_seq = index + i
                max_idx = len(self.data["dsf"])-1
                if index_seq > max_idx: index_seq = max_idx

                feature_array = self.data["dsf"][index_seq]
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf, dsf_seq], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current ... future]
        else:
            last_valid_idx = audio_id
            for i in range(1,self.opt.seq_len):
                index_seq = index - i
                if index_seq < 0: index_seq = 0

                feature_array = self.data["dsf"][index_seq]
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf_seq, dsf], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current]


        #################################
        weight = 1.0 / self.n_frames_total
        
        return {#'TARGET': TARGET, 'UV': UV,
                'paths': '', #img_path,
                'intrinsics': np.array(intrinsics),
                'extrinsics': np.array(extrinsics),
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
