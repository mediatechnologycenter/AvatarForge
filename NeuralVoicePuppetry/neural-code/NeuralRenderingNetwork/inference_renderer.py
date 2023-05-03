import copy
import numpy as np
import torch
import torchvision.transforms as transforms
import sys
import os
import cv2

from tqdm import tqdm
from scipy.ndimage import binary_fill_holes
from scipy.ndimage import binary_erosion

from data.custom_aligned_dataset import CustomAlignedDataset
from options.test_options import TestOptions
from models import create_model
from util.visualizer import InferenceVisualizer

# deca import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third/DECA/')))
from decalib.deca import *
from decalib.utils.config import cfg as deca_cfg

class InferenceManager:

    def __init__(self, opt):
        self.opt = opt

        deca_cfg.model.use_tex = False
        self.deca = DECA(config=deca_cfg, device="cuda" if torch.cuda.is_available() else "cpu")

        self.model = create_model(opt)
        self.model.setup(opt)

        # self.new_expressions = np.loadtxt(opt.expr_path)
        # print(self.new_expressions.shape)

        if opt.textureModel == 'DynamicNeuralTextureAudio':
            self.load_source = True
            opt_source = copy.copy(opt)
            opt_source.dataroot = opt.source_dataroot
            self.dataset_source = CustomAlignedDataset()
            self.dataset_source.initialize(opt_source)
            opt_target = copy.copy(opt)
            opt_target.dataroot = opt.target_dataroot
            self.dataset_target = CustomAlignedDataset()
            self.dataset_target.initialize(opt_target)

        else:
            self.load_source = False
            opt_target = copy.copy(opt)
            opt_target.dataroot = opt.target_dataroot
            self.dataset_target = CustomAlignedDataset()
            self.dataset_target.initialize(opt_target)
            opt_source = copy.copy(opt)
            opt_source.dataroot = opt.source_dataroot
            self.dataset_source = CustomAlignedDataset()
            self.dataset_source.initialize(opt_source)


        self.frame_id_source = opt.frame_id_source
        self.frame_id_target = opt.frame_id_target
        print("Running inference with the following config:")
        print("SOURCE: frame {} in {}".format(self.frame_id_source, opt.source_dataroot))
        print("TARGET: frame {} in {}".format(self.frame_id_target, opt.target_dataroot))

        self.visualizer = InferenceVisualizer(opt)


    def get_codedict(self, frame_id):

        codedict = torch.load(os.path.join(self.opt.dataroot, 'DECA_codedicts', f'codedict_{frame_id}.pt'))

        return codedict

    def get_expr(self, frame_id):

        expr = np.load(os.path.join(self.opt.expr_path, f'expr_{frame_id}.npy'))[0]

        return expr

    def preprocess_uv(self, uv):
        IMG_DIM_Y = 256
        IMG_DIM_X = 256

        if uv.shape[0] != IMG_DIM_Y or uv.shape[1] != IMG_DIM_X:
            # print("UV needs resizing from {} to {},{}".format(uv.shape, self.IMG_DIM_Y, self.IMG_DIM_X))
            uv = cv2.resize(uv, (IMG_DIM_Y, IMG_DIM_X))

        # issue only happens with multiple concurrent reads
        if not (-1 <= uv.min() and uv.max() <= 1):
            print("frame invalid", frame_id)

        assert -1 <= uv.min() and uv.max() <= 1, "UV not in range [-1, 1]! min: {} max: {}".format(uv.min(), uv.max())
        uv_tensor = transforms.ToTensor()(uv.astype(np.float32))

        return uv_tensor.cuda()

    def preprocess_deca_details(self, deca_details):
        IMG_DIM_Y = 256
        IMG_DIM_X = 256

        if deca_details.shape[0] != IMG_DIM_Y or deca_details.shape[1] != IMG_DIM_X:
            # print("UV needs resizing from {} to {},{}".format(uv.shape, self.IMG_DIM_Y, self.IMG_DIM_X))
            deca_details = cv2.resize(deca_details, (IMG_DIM_Y, IMG_DIM_X))

        deca_details_tensor = transforms.ToTensor()(deca_details.astype(np.float32))

        return deca_details_tensor.cuda()

    def preprocess_mask(self, mask):
        IMG_DIM_Y = 256
        IMG_DIM_X = 256

        if mask.shape[0] != IMG_DIM_Y or mask.shape[1] != IMG_DIM_X:
            # print("UV needs resizing from {} to {},{}".format(uv.shape, self.IMG_DIM_Y, self.IMG_DIM_X))
            mask = cv2.resize(mask, (IMG_DIM_Y, IMG_DIM_X))

        mask = mask>0
        mask = binary_fill_holes(mask).astype(np.float32)
        mask = binary_erosion(mask, iterations=8)
        mask_tensor = transforms.ToTensor()(mask.astype(np.float32))

        return mask_tensor.cuda()

    def run_frame(self, frame_id_source, frame_id_target):

        new_expr = self.get_expr(frame_id_source)
        codedict = self.get_codedict(frame_id_target)

        # Render new UV map with shape and pose from target, and expression from source
        exp = torch.tensor(new_expr[:-3], dtype=torch.float32).cuda().unsqueeze(0)

        pose = codedict['pose'][:, :-3]
        jaw_pose = torch.tensor(new_expr[-3:], dtype=torch.float32).cuda().unsqueeze(0)
        new_pose = torch.cat((pose, jaw_pose), dim=1).cuda()

        if self.opt.deca_details:
            uv, deca_details = self.deca.render_uv_details(codedict, exp, new_pose)
            mask = self.deca.render_mask(uv)
            mask = self.preprocess_mask(mask.cpu().detach().numpy())
            uv = self.preprocess_uv(uv[0].cpu().numpy())
            deca_details = self.preprocess_deca_details(deca_details[0].permute(1, 2, 0).cpu().detach().numpy())

            uv = uv * mask
            deca_details = deca_details * mask

        else:
            uv = self.deca.render_uv(codedict, exp, new_pose)[0]
            uv = self.preprocess_uv(uv.cpu().numpy())

        if self.load_source:
            source = self.dataset_source[frame_id_source]
            target = self.dataset_target[frame_id_target]

            item = source

            item["SOURCE"] = source['TARGET'].cuda().unsqueeze(0)
            item["TARGET"] = target['TARGET'].cuda().unsqueeze(0)
            item["UV"] = uv.cuda().unsqueeze(0)
            item["expressions"] = torch.tensor(new_expr).cuda().unsqueeze(0)
            item["audio_deepspeech"] = item["audio_deepspeech"].cuda().unsqueeze(0)

        else:
            target = self.dataset_target[frame_id_target]

            item = target

            item["SOURCE"] = target['TARGET'].cuda().unsqueeze(0)
            item["TARGET"] = target['TARGET'].cuda().unsqueeze(0)
            item["UV"] = uv.cuda().unsqueeze(0)
            item["expressions"] = torch.tensor(new_expr).cuda().unsqueeze(0)
            item["audio_deepspeech"] = item["audio_deepspeech"].cuda().unsqueeze(0)

        if self.opt.deca_details:
            item["deca_details"] = deca_details.cuda().unsqueeze(0)

        self.model.set_input(item)
        self.model.forward()
        results = self.model.get_current_visuals()

        # results = None

        return results


    def run_inference(self):

        # Run inference for all frames
        try:
            n_frames = min(len(self.dataset_source), len(self.dataset_target))

        except AttributeError:
            n_frames = len(self.dataset_target)

        if self.frame_id_source >= 0 and self.frame_id_target >= 0:

            for frame_id in tqdm(range(self.frame_id_source, self.frame_id_target)):
                results = self.run_frame(frame_id, frame_id)
                self.visualizer.reset()
                self.visualizer.display_current_results(results, frame_id, save_result=True)

        else:

            for frame_id in tqdm(range(n_frames)):
                results = self.run_frame(frame_id, frame_id)
                self.visualizer.reset()
                self.visualizer.display_current_results(results, frame_id, save_result=True)


if __name__ == '__main__':
    # training dataset
    opt = TestOptions().parse()
    opt.serial_batches = True
    opt.display_id = 0

    # model
    manager = InferenceManager(opt)
    manager.run_inference()



