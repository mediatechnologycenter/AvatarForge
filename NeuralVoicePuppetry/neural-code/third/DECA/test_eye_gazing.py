import sys
import os
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../third/DECA/')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils.config import cfg as deca_cfg

def save_img(name, image):
    to_save = image[0].permute(1, 2, 0).cpu().numpy()
    to_save = to_save * 255.
    to_save = to_save.astype(np.uint8)
    filename = name + '.jpg'
    img = Image.fromarray(to_save)
    img.save(filename)

deca_cfg.model.use_tex = False
device = "cuda" if torch.cuda.is_available() else "cpu"
deca = DECA(config=deca_cfg, device=device)

# load test images
testdata = datasets.TestData('test_expression.jpg', iscrop=True, face_detector='fan')
images = testdata[0]['image'].to(device)[None, ...]
codedict = deca.encode(images)
expr = codedict['exp']

# load test images
testdata = datasets.TestData('test_gaze.jpg', iscrop=True, face_detector='fan')
images = testdata[0]['image'].to(device)[None, ...]

codedict = deca.encode(images)
opdict, visdict = deca.decode(codedict)

save_img('shape', visdict['shape_images'])

opdict, visdict = deca.decode(codedict)
verts_og = opdict['vertices']
gird_og = opdict['grid']
texture_og = opdict['uv_texture_gt']
image = F.grid_sample(texture_og, gird_og, align_corners=False)
save_img('rendered_image_og', image)


# codedict['exp'] = expr
opdict, visdict = deca.decode_eyes(codedict)
verts_gaze = opdict['vertices']
gird_gaze = opdict['grid']
texture_gaze = opdict['uv_texture_gt']
image = F.grid_sample(texture_og, gird_gaze, align_corners=False)
save_img('rendered_image_gaze', image)


euclidena_dist = sum(((verts_og - verts_gaze)**2).flatten())
print('Mesh distance: ', euclidena_dist)

gird_dist = sum(((gird_og - gird_gaze)**2).flatten())
print('Grid distance: ', gird_dist)

text_dist = sum(((texture_og - texture_gaze)**2).flatten())
print('Texture distance: ', text_dist)
