import os.path
import random
import torchvision.transforms as transforms
import torch
import numpy as np
from .base_dataset import BaseDataset
from .audio import Audio
#from data.image_folder import make_dataset
from PIL import Image
import h5py

def make_dataset(dir):
    images = []
    ids = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['.deepspeech.npy']):
                id_str = fname[:-15]
                i = int(id_str)
                ids.append(i)
    ids = sorted(ids)

    for id in ids:
        fname=str(id)+'.deepspeech.npy'
        path = os.path.join(root, fname)
        images.append(path)
    return images

def make_ids(paths, root_dir):
    ids = []
    
    for fname in paths:
        l = fname.rfind('/')
        #id_str = fname[l+1:-4]
        id_str = fname[l+1:-15]
        i = int(id_str)
        #print(fname, ': ', i)
        ids.append(i)
    return ids

class AudioDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.audiofeat_path = os.path.join(opt.dataroot, 'audio_feature')

        self.audiofeat_list = sorted(os.listdir(self.audiofeat_path))

        print('\taudio_feat_path:', self.audiofeat_path)

        opt.nObjects = 1
        opt.nTrainObjects = 116 # TODO
        opt.nTestObjects = 1
        opt.test_sequence_names = [[opt.dataroot.split("/")[-1], 'test']]
        assert(opt.resize_or_crop == 'resize_and_crop')

        if opt.isTrain:
            print('ERROR: audio_dataset only allowed for test')
            exit()

    def getSampleWeights(self):
        weights = np.ones((len(self.frame_paths)))
        return weights

    def getAudioFilename(self):
        return os.path.join(self.root, 'audio.wav')

    def getAudioFeatureFilename(self, idx):
        return self.frame_paths[idx % len(self.frame_paths)]

    def load_npy(self, index):

        file_path = os.path.join(self.audiofeat_path, "%05d.deepspeech.npy" % index)

        feature_array = np.load(file_path)

        return feature_array

    def __getitem__(self, index):

        # load deepspeech feature
        feature_array = self.load_npy(index)
        dsf_np = np.resize(feature_array,  (16,29,1))
        dsf = transforms.ToTensor()(dsf_np.astype(np.float32))

        # load sequence data if necessary
        if self.opt.look_ahead:# use prev and following frame infos
            r = self.opt.seq_len//2
            for i in range(1,r): # prev frames
                index_seq = index - i
                if index_seq < 0: index_seq = 0

                feature_array = self.load_npy(index_seq)
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf_seq, dsf], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current]
            
            for i in range(1,self.opt.seq_len - r + 1): # following frames
                index_seq = index + i
                max_idx = len(self.audiofeat_list)-1
                if index_seq > max_idx: index_seq = max_idx

                feature_array = self.load_npy(index_seq)
                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf, dsf_seq], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current ... future]
        else:
            for i in range(1, self.opt.seq_len):
                index_seq = index - i
                if index_seq < 0: index_seq = 0

                feature_array = self.load_npy(index_seq)

                dsf_np = np.resize(feature_array,  (16,29,1))
                dsf_seq = transforms.ToTensor()(dsf_np.astype(np.float32)) # 1 x 16 x 29
                dsf = torch.cat([dsf_seq, dsf], 0)  # seq_len x 16 x 29
                # note the ordering [old ... current]


        #################################
        zeroIdentity = torch.zeros(100)
        zeroExpressions = torch.zeros(76)

        target_id = -1
        internal_sequence_id = 0

        weight = 1.0 / self.__len__()

        return {'paths': '',
                'expressions': zeroExpressions,
                'identity': zeroIdentity,
                'intrinsics': np.zeros((4)),
                'extrinsics': np.zeros((4,4)),
                'audio_deepspeech': dsf, # deepspeech feature
                'target_id':target_id,
                'internal_id':internal_sequence_id,
                'weight': np.array([weight]).astype(np.float32)}

    def __len__(self):
        return len(self.audiofeat_list)

    def name(self):
        return 'AudioDataset'
