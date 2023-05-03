import os.path

import h5py
import random
import torchvision.transforms as transforms
import torch
import numpy as np

# from mutil.bfm2017 import BFM2017
# from mutil.np_util import adjust_width

from data.base_dataset import BaseDataset
from data.audio import Audio
import cv2
from scipy.ndimage.morphology import binary_fill_holes
from scipy.ndimage.morphology import binary_erosion
#from data.image_folder import make_dataset
from PIL import Image

#def make_dataset(dir):
#    images = []
#    assert os.path.isdir(dir), '%s is not a valid directory' % dir
#    for root, _, fnames in sorted(os.walk(dir)):
#        for fname in fnames:
#            if any(fname.endswith(extension) for extension in ['.bin', '.BIN']):
#                path = os.path.join(root, fname)
#                images.append(path)
#    return sorted(images)
# from mutil.pytorch_utils import to_tensor


def make_dataset(dir):
    images = []
    ids = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in ['.bin', '.BIN']):
                id_str = fname[:-4]
                i = int(id_str)
                ids.append(i)
    ids = sorted(ids)

    for id in ids:
        fname=str(id)+'.bin'
        path = os.path.join(root, fname)
        images.append(path)
    return images

def load_intrinsics(input_dir):
    file = open(input_dir+"/intrinsics.txt", "r")
    intrinsics = [[(float(x) for x in line.split())] for line in file]
    file.close()
    intrinsics = list(intrinsics[0][0])
    return intrinsics

def load_rigids(input_dir):
    file = open(input_dir+"/rigid.txt", "r")
    rigid_floats = [[float(x) for x in line.split()] for line in file] # note that it stores 5 lines per matrix (blank line)
    file.close()
    all_rigids = [ [rigid_floats[4*idx + 0],rigid_floats[4*idx + 1],rigid_floats[4*idx + 2],rigid_floats[4*idx + 3]] for idx in range(0, len(rigid_floats)//4) ]
    return all_rigids

def load_expressions(input_dir):
    file = open(input_dir+"/expression.txt", "r")
    expressions = [[float(x) for x in line.split()] for line in file]
    file.close()
    return expressions

def load_audio(input_dir):
    audio = Audio(input_dir + '/audio.mp3', write_mel_spectogram = False)
    #audio = Audio(input_dir + '/audio.mp3', write_mel_spectogram = True)

    return audio 

def make_ids(paths, root_dir):
    ids = []
    
    for fname in paths:
        l = fname.rfind('/')
        id_str = fname[l+1:-4]
        i = int(id_str)
        #print(fname, ': ', i)
        ids.append(i)
    return ids

class CustomAlignedDataset(BaseDataset):
    IMG_DIM_Y = 256
    IMG_DIM_X = 256

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.data = h5py.File(self.root, 'r')
        print(self.data.keys())
        # self.audio = load_audio(self.data_dir)
        # self.audio_window_size = opt.audio_window_size

        opt.nObjects = 1

    def preprocess_image(self, frame_id):
        img = self.data["frame"][frame_id]
        img = img.astype(np.float) / 255
        if img.shape[0] != self.IMG_DIM_Y or img.shape[1] != self.IMG_DIM_X:
            # print("Image needs resizing from {} to {},{}".format(img.shape, self.IMG_DIM_Y, self.IMG_DIM_X))
            img = cv2.resize(img, (self.IMG_DIM_Y, self.IMG_DIM_X))  # TODO: np.resize vs. cv2.resize

        assert 0 <= img.min() and img.max() <= 1
        img_tensor = transforms.ToTensor()(img.astype(np.float32))
        img_tensor = 2.0 * img_tensor - 1.0
        return img_tensor

    def preprocess_uv(self, frame_id):
        uv = self.data["uv"][frame_id]  # TODO: how to store uv? Do we need the same dims?

        if uv.shape[0] != self.IMG_DIM_Y or uv.shape[1] != self.IMG_DIM_X:
            # print("UV needs resizing from {} to {},{}".format(uv.shape, self.IMG_DIM_Y, self.IMG_DIM_X))
            uv = cv2.resize(uv, (self.IMG_DIM_Y, self.IMG_DIM_X))  # TODO: np.resize vs. cv2.resize

        # issue only happens with multiple concurrent reads
        if not (-1 <= uv.min() and uv.max() <= 1):
            print("frame invalid", frame_id)
        uv_tensor = transforms.ToTensor()(uv.astype(np.float32))

        assert -1 <= uv.min() and uv.max() <= 1, "UV not in range [-1, 1]! min: {} max: {}".format(uv.min(), uv.max())
        return uv_tensor

    def preprocess_expressions(self, frame_id):
        return torch.tensor(self.data['ep'][frame_id])

    def preprocess_audio_features(self, frame_id):
        # load deepspeech feature
        feature_array = self.data["dsf"][frame_id]
        assert feature_array.shape == (16, 29)
        dsf_np = np.expand_dims(feature_array, 2)
        dsf_tensor = transforms.ToTensor()(dsf_np.astype(np.float32))
        return dsf_tensor

    def preprocess_deca_details(self, frame_id):
        deca_details = self.data["deca_details"][frame_id]  # TODO: how to store uv? Do we need the same dims?

        if deca_details.shape[0] != self.IMG_DIM_Y or deca_details.shape[1] != self.IMG_DIM_X:
            # print("UV needs resizing from {} to {},{}".format(uv.shape, self.IMG_DIM_Y, self.IMG_DIM_X))
            deca_details = cv2.resize(deca_details, (self.IMG_DIM_Y, self.IMG_DIM_X))  # TODO: np.resize vs. cv2.resize

        deca_details_tensor = transforms.ToTensor()(deca_details.astype(np.float32))
        return deca_details_tensor

    def preprocess_mask(self, frame_id):
        mask = self.data["mask"][frame_id]  # TODO: how to store uv? Do we need the same dims?

        if mask.shape[0] != self.IMG_DIM_Y or mask.shape[1] != self.IMG_DIM_X:
            # print("UV needs resizing from {} to {},{}".format(uv.shape, self.IMG_DIM_Y, self.IMG_DIM_X))
            mask = cv2.resize(mask, (self.IMG_DIM_Y, self.IMG_DIM_X))

        mask = mask>0
        mask = binary_fill_holes(mask).astype(np.float32)
        mask = binary_erosion(mask, iterations=8)
        mask_tensor = transforms.ToTensor()(mask.astype(np.float32))

        return mask_tensor

    def __getitem__(self, index):
        frame_id = index
        # expressions
        expression_tensor = self.preprocess_expressions(frame_id)
        img_tensor = self.preprocess_image(frame_id)
        uv_tensor = self.preprocess_uv(frame_id)
        mask_tensor = self.preprocess_mask(frame_id)
        dsf_tensor = self.preprocess_audio_features(frame_id)

        if self.opt.deca_details:
            deca_details_tensor = self.preprocess_deca_details(frame_id)

        #################################
        ####### apply augmentation ######
        #################################
        if not self.opt.no_augmentation:
            # random dimensions
            new_dim_x = np.random.randint(int(self.IMG_DIM_X * 0.75), self.IMG_DIM_X+1)
            new_dim_y = np.random.randint(int(self.IMG_DIM_Y * 0.75), self.IMG_DIM_Y+1)
            new_dim_x = int(np.floor(new_dim_x / 64.0) * 64 ) # << dependent on the network structure !! 64 => 6 layers
            new_dim_y = int(np.floor(new_dim_y / 64.0) * 64 )
            if new_dim_x > self.IMG_DIM_X: new_dim_x -= 64
            if new_dim_y > self.IMG_DIM_Y: new_dim_y -= 64

            # random pos
            if self.IMG_DIM_X == new_dim_x: offset_x = 0
            else: offset_x = np.random.randint(0, self.IMG_DIM_X-new_dim_x)
            if self.IMG_DIM_Y == new_dim_y: offset_y = 0
            else: offset_y = np.random.randint(0, self.IMG_DIM_Y-new_dim_y)

            # select subwindow
            img_tensor = img_tensor[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]
            uv_tensor = uv_tensor[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]

            # compute new intrinsics
            # TODO: atm not needed but maybe later

        else:
            new_dim_x = int(np.floor(self.IMG_DIM_X / 64.0) * 64 ) # << dependent on the network structure !! 64 => 6 layers
            new_dim_y = int(np.floor(self.IMG_DIM_Y / 64.0) * 64 )
            offset_x = 0
            offset_y = 0
            # select subwindow
            img_tensor = img_tensor[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]
            uv_tensor = uv_tensor[:, offset_y:offset_y+new_dim_y, offset_x:offset_x+new_dim_x]

            # compute new intrinsics
            # TODO: atm not needed but maybe later

        #################################
        return_dict = {'TARGET': img_tensor,
                'UV': uv_tensor * mask_tensor,
                # 'paths': self.frame_paths[index],#img_path,
                # 'intrinsics': intrinsics,
                # 'extrinsics': extrinsics,
                'expressions': expression_tensor,
                # 'audio_mels': mels,
                'audio_deepspeech': dsf_tensor, # deepspeech feature
        }

        if self.opt.deca_details:
            return_dict['deca_details'] = deca_details_tensor * mask_tensor

        return return_dict

    def __len__(self):
        return self.data["dsf"].shape[0]

    def name(self):
        return 'CustomAlignedDataset'

