# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from logging import raiseExceptions
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
import numpy as np

def readFlow(fn):
    """ Read .flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'fn = %s'%(fn)
    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            # print 'Reading %d x %d flo file\n' % (w, h)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            # Reshape data into 3D array (columns, rows, bands)
            # The reshape here is for visualization, the original code is (w,h,2)
            return np.resize(data, (int(h), int(w), 2))

# Example from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class CustomDataset(Dataset):
    def __init__(self, args, input_root_dir, output_root_dir, label_root_dir, transform_input=None, transform_output=None, transform_label_map=None, mode='train'):
        """
        Args:
            input_root_dir (string): Directory with all the input images.
            output_root_dir (string): Directory with all the output images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.use_label_maps = args.use_label_maps
        self.mode = mode
        self.input_root_dir = input_root_dir
        self.input_img_name_list = sorted(os.listdir(self.input_root_dir))
        self.transform_input = transform_input
        if mode=='train' or mode=='val':
            self.output_root_dir = output_root_dir
            self.output_img_name_list = sorted(os.listdir(self.output_root_dir))
            self.transform_output = transform_output
            if args.use_label_maps:
                self.transform_label_map = transform_label_map
                self.label_root_dir = label_root_dir
                self.label_img_name_list = sorted(os.listdir(self.label_root_dir))

    def convert_to_rgb(self, image):
            if image.mode == 'RGBA':
                image.load() 
                image_new = Image.new("RGB", image.size, (255, 255, 255))
                image_new.paste(image, mask=image.split()[3])
            elif image.mode == 'RGB' or image.mode == 'L':
                image_new = image
            else:
                raiseExceptions('Non-compatible image format!')
            return image_new

    def __len__(self):
        return len(os.listdir(self.input_root_dir))

    def __getitem__(self, idx):

        input_img_name = os.path.join(self.input_root_dir,
                                self.input_img_name_list[idx])
        input_image = Image.open(input_img_name)
        input_image = self.convert_to_rgb(input_image)

        if self.mode == 'train' or self.mode == 'val':
            output_img_name = os.path.join(self.output_root_dir,
                                           self.output_img_name_list[idx])
            output_image = Image.open(output_img_name)
            output_image = self.convert_to_rgb(output_image)

            if self.use_label_maps:
                label_img_name = os.path.join(self.label_root_dir,
                                              self.label_img_name_list[idx])
                label_image = Image.open(label_img_name)
                label_image = torch.unsqueeze(torch.tensor(np.array(self.convert_to_rgb(label_image))), 0)


                sample = {'input_image': self.transform_input(input_image), 
                        'output_image': self.transform_output(output_image), 
                        'label_image': self.transform_label_map(label_image), 
                        'name': self.input_img_name_list[idx]}
            
            else:
                sample = {'input_image': self.transform_input(input_image), 
                         'output_image': self.transform_output(output_image),  
                         'name': self.input_img_name_list[idx]}

        else:
            sample = {'input_image': self.transform_input(input_image), 'name': self.input_img_name_list[idx]}

        return sample

class OpticalFlowDataset(Dataset):
    def __init__(self, args, input_root_dir, flow_dir, output_root_dir, transform_input=None, transform_output=None, mode='train'):
        """
        Args:
            input_root_dir (string): Directory with all the input images.
            flow_dir(string): Directory with all the computed optical flow.
            output_root_dir (string): Directory with all the output images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
        """
        self.use_label_maps = args.use_label_maps
        self.mode = mode
        self.input_root_dir = input_root_dir
        self.input_img_name_list = sorted(os.listdir(self.input_root_dir))
        self.transform_input = transform_input

        self.optical_steps = args.optical_steps

        if mode=='train' or mode=='val':
            self.output_root_dir = output_root_dir
            self.output_img_name_list = sorted(os.listdir(self.output_root_dir))
            self.transform_output = transform_output

            self.flow_dir = flow_dir
            self.flow_name_list = sorted(os.listdir(self.flow_dir))


    def convert_to_rgb(self, image):
            if image.mode == 'RGBA':
                image.load() 
                image_new = Image.new("RGB", image.size, (255, 255, 255))
                image_new.paste(image, mask=image.split()[3])
            elif image.mode == 'RGB' or image.mode == 'L':
                image_new = image
            else:
                raiseExceptions('Non-compatible image format!')
            return image_new

    def __len__(self):
        return len(os.listdir(self.input_root_dir))

    def __getitem__(self, idx):

        input_img_name = os.path.join(self.input_root_dir,
                                self.input_img_name_list[idx])
        input_image = Image.open(input_img_name)
        input_image = self.convert_to_rgb(input_image)

        if self.mode == 'train' or self.mode == 'val':
            # Load output image
            output_img_name = os.path.join(self.output_root_dir,
                                           self.output_img_name_list[idx])
            output_image = Image.open(output_img_name)
            output_image = self.convert_to_rgb(output_image)

            # Load previous images & Optical Flows
            flow = np.zeros_like(readFlow(os.path.join(self.flow_dir, self.flow_name_list[0])))

            flows = [] 
            prev_imgs = []  
            
            if idx > self.optical_steps:
                
                for i in range(self.optical_steps):
                    # Load prev image
                    prev_img_name = os.path.join(self.output_root_dir,
                                            self.output_img_name_list[idx - i])
                    prev_img = Image.open(prev_img_name)
                    prev_img = self.convert_to_rgb(prev_img)
                    prev_imgs.append(self.transform_output(prev_img))

                    # load flow
                    flow_name = os.path.join(self.flow_dir,
                                            self.flow_name_list[idx - i])
                    tmp_flow = readFlow(flow_name)
                    flow += tmp_flow
                    flows.append(self.transform_output(flow))
            
            else:
                prev_img_name = os.path.join(self.output_root_dir, self.output_img_name_list[idx])
                prev_img = Image.open(prev_img_name)
                prev_img = self.convert_to_rgb(prev_img)
                flows = [self.transform_output(flow)] * self.optical_steps
                prev_imgs = [self.transform_output(prev_img)] * self.optical_steps
            
            sample = {'input_image': self.transform_input(input_image), 
                      'output_image': self.transform_output(output_image),  
                      'prev_images': prev_imgs,
                      'flows': flows,
                      'name': self.input_img_name_list[idx]}

        else:
            sample = {'input_image': self.transform_input(input_image), 'name': self.input_img_name_list[idx]}

        return sample
