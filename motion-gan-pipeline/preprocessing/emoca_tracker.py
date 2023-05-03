# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from gdl_apps.EmotionRecognition.utils.io import load_model
from gdl.datasets.ImageTestDataset import TestData
import gdl
import numpy as np
import os
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch.functional import F
from gdl.datasets.AffectNetDataModule import AffectNetExpressions
from gdl.utils.other import get_path_to_assets
from tqdm import tqdm

def load_dir(lmspath, framepath, start, end):
    lmss = []
    imgs_paths = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(lmspath, str(i) + '.lms')):
            lms = np.loadtxt(os.path.join(
                lmspath, str(i) + '.lms'), dtype=np.float32)
            lmss.append(lms)
            imgs_paths.append(os.path.join(framepath, str(i) + '.jpg'))
    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).cuda()
    return imgs_paths

class EMOCA_tracker:
    def __init__(self):
            
        model_name = 'ResNet50'
        path_to_models = get_path_to_assets() /"EmotionRecognition"

        path_to_models = path_to_models / "image_based_networks"

        self.model = load_model(Path(path_to_models) / model_name)
        print(self.model)
        self.model.cuda()
        self.model.eval()

    def __call__(self, images, tform=None):

        codedict = self.model(images)

        return codedict
    
    def save_images(self, batch, predictions, output_folder):
        # Save the images

        softmax = F.softmax(predictions["expr_classification"])
        top_expr =  torch.argmax(softmax, dim=1)
        for i in range(len(batch["image"])):
            img = batch["image"][i].cpu().detach().numpy()
            img = img.transpose(1, 2, 0)
            img = img * 255
            img = img.astype(np.uint8)

            plt.figure()
            # plot the image with matplotlib 
            plt.imshow(img)
            # write valence and arousal to the image
            expr = AffectNetExpressions(int(top_expr[i].item()))
            text = "Predicted emotion:\n"
            text += f'Arousal: {predictions["arousal"][i].item():.2f} \nValence: {predictions["valence"][i].item():.2f}'
            text += f"\nExpression: {expr.name}, {softmax[i][expr.value].item()*100:.2f}%"
            plt.title(text)
            out_fname = Path(output_folder) / f"{batch['image_name'][i]}.png"
            # save the image to the output folder
            
            # axis off 
            plt.axis('off')
            plt.savefig(out_fname)
            plt.close()


def emotion_detection(dataset_base, emotion_dir):
    '''
        Face tracker using FLAME model.
        Used to have geometry prior for nerf sampling.
        '''

    id_dir = dataset_base
    debug_emotions = os.path.join(id_dir, 'debug', 'emotions_imgs')
    Path(debug_emotions).mkdir(parents=True, exist_ok=True)

    emoca_tracker = EMOCA_tracker()

    # Run deca on all frames
    testdata = TestData(os.path.join(id_dir, 'frames'), face_detector="fan", max_detection=20)
    
    for i, data in enumerate(tqdm(testdata)):
        batch = testdata[i]
        batch["image"] = batch["image"].cuda()
        predictions = emoca_tracker(batch)
        npy_pred = {k: v.cpu().detach().numpy() for k,v in predictions.items()}
        np.save(os.path.join(emotion_dir, '%5d.npy' % i), npy_pred)

        emoca_tracker.save_images(batch, predictions, debug_emotions)

if __name__ == '__main__':

    dataset_base = '/media/apennino/EmotionDetection/Test/Greta/'
    emotion_dir = '/media/apennino/EmotionDetection/Test/Greta/emotions/'
    emotion_detection(dataset_base, emotion_dir)
