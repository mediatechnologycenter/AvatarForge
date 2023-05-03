import PIL
import torch
from torchvision import transforms
import wandb
import configargparse
from pathlib import Path
import os

def create_image_pair(images_to_concat):
    
    num_images = images_to_concat[0].shape[0]
    images = []
    for image_num in range(num_images):
        image = images_to_concat[0][image_num,::]
        if image.shape[0] == 1:
            image = torch.cat((image,)*3, axis=0)
        for img_num, img in enumerate(images_to_concat):
            if not img_num==0:
                image_to_concat = img[image_num,::]
                if image_to_concat.shape[0] == 1:
                    image_to_concat = torch.cat((image_to_concat,)*3, axis=-1)
                image = torch.cat((image, image_to_concat), dim=2)
        image_transform = transforms.ToPILImage()
        image = image_transform(image)
        images.append(image)
    return images


def save_image_list(image_list, save_path, names):
    for image, name in zip(image_list, names):
        image.save(save_path+name)
