import os.path

import h5py
import random
import torchvision.transforms as transforms
import torch
import numpy as np

from mutil.bfm2017 import BFM2017
from mutil.np_util import adjust_width

from data.base_dataset import BaseDataset
from data.audio import Audio
import cv2
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.structures import Meshes
#from data.image_folder import make_dataset
from PIL import Image

from data.custom_aligned_dataset import CustomAlignedDataset


class CustomAlignedInferenceDataset(CustomAlignedDataset):
    def initialize(self, opt):
        super().initialize(opt)
        self.data_source = h5py.File(opt.dataroot_source, 'r')

        self.bfm = BFM2017(opt.path_bfm)
        self.uv_renderer = None

    def create_uv_renderer(self, height):
        from mutil.renderer import BFMUVRenderer
        R, T = look_at_view_transform(eye=((0, 0, 0),), at=((0, 0, -1),), up=((0, 1, 0),))
        self.uv_renderer = BFMUVRenderer(self.opt.path_uv, device="cuda" if torch.cuda.is_available() else "cpu", fov=63, image_size=height, R=R, T=T)


    def render_uv(self, mesh, height, width):
        if self.uv_renderer is None:
            # We do this because we only now the height now.
            self.uv_renderer = self.create_uv_renderer(height)

        rendered_uv_np = self.uv_renderer.render(mesh)
        rendered_uv_np = adjust_width(rendered_uv_np, width, is_uv=True)
        return rendered_uv_np

    def __getitem__(self, index):
        item = super().__getitem__(index)
        frame_id = index

        # # Expressions
        # source_expressions_tensor = torch.tensor(self.data_source['ep'][frame_id])
        # item['expressions'] = source_expressions_tensor
        #
        # # Audio features
        # feature_array = self.data_source["dsf"][frame_id]
        # dsf_np = np.expand_dims(feature_array, 2)
        # dsf_tensor = transforms.ToTensor()(dsf_np.astype(np.float32))
        # item['audio_deepspeech'] = dsf_tensor  # deepspeech feature
        #
        # # image and UV texture map (rendered)
        # new_dim_x = int(np.floor(self.IMG_DIM_X / 64.0) * 64)  # << dependent on the network structure !! 64 => 6 layers
        # new_dim_y = int(np.floor(self.IMG_DIM_Y / 64.0) * 64)
        # img_source = self.data_source["frame"][frame_id]
        # img_tensor_source = self.preprocess_image(img_source)
        # item['SOURCE'] = img_tensor_source

        shape_coeff = self.data['sp'][frame_id]
        expression_coeff = self.data_source['ep'][frame_id]
        verts = self.bfm.generate_vertices(shape_coeff, expression_coeff)
        R, t, s = self.data['R'][frame_id], self.data['t'][frame_id], self.data['s'][frame_id]
        verts = self.bfm.apply_Rts(verts, R, t, s)
        mesh = Meshes([verts], [self.bfm.faces])

        # uv = self.render_uv(mesh, img_source.shape[0], img_source.shape[1])
        # uv_tensor = transforms.ToTensor()(uv.astype(np.float32))
        # uv_tensor = uv_tensor[:, 0: new_dim_y, 0:new_dim_x]
        # item['UV'] = uv_tensor
        return item