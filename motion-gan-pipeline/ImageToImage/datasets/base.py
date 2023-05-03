# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from .custom_datasets import CustomDataset, OpticalFlowDataset
from torchvision.transforms.functional import InterpolationMode


def build_dataset(args, mode):

    if args.dataset_type == "CustomDataset":
        transform_input = transforms.Compose([transforms.ToTensor(), transforms.Resize([args.height, args.width]),]) # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        transform_output = transforms.Compose([transforms.ToTensor(), transforms.Resize([args.height, args.width]),]) 
        transform_label_map = transforms.Compose([transforms.Resize([args.height, args.width], interpolation=InterpolationMode.NEAREST),]) 
        if mode=='train':        
            input_dir = args.input_train_root_dir
            output_dir = args.output_train_root_dir
            label_dir = args.label_train_root_dir
        elif mode=='val':
            input_dir = args.input_val_root_dir
            output_dir = args.output_val_root_dir
            label_dir = args.label_val_root_dir
        elif mode=='test':
            input_dir = args.input_test_root_dir
            output_dir = None
            label_dir = None

        dataset = CustomDataset(args = args,
                                input_root_dir=input_dir, 
                                output_root_dir=output_dir, 
                                label_root_dir=label_dir, 
                                transform_input=transform_input, 
                                transform_output=transform_output, 
                                transform_label_map=transform_label_map, 
                                mode=mode)

    elif args.dataset_type == "OpticalFlowDataset":
        transform_input = transforms.Compose([transforms.ToTensor(), transforms.Resize([args.height, args.width]),]) # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
        transform_output = transforms.Compose([transforms.ToTensor(), transforms.Resize([args.height, args.width]),]) 

        if mode=='train':        
            input_dir = args.input_train_root_dir
            output_dir = args.output_train_root_dir
            flow_dir = args.flow_train_root_dir
        elif mode=='val':
            input_dir = args.input_val_root_dir
            output_dir = args.output_val_root_dir
            flow_dir = args.flow_val_root_dir
        elif mode=='test':
            input_dir = args.input_test_root_dir
            output_dir = None
            flow_dir = None

        dataset = OpticalFlowDataset(args = args,
                                    input_root_dir=input_dir, 
                                    flow_dir=flow_dir,
                                    output_root_dir=output_dir, 
                                    transform_input=transform_input, 
                                    transform_output=transform_output, 
                                    mode=mode)
    else:
        raise NotImplementedError()

    return dataset


def build_dataloader(args, mode, shuffle):

    dataset = build_dataset(args, mode)    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=False, drop_last=True)

    return dataloader
