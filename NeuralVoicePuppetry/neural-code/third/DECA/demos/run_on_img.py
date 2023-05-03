
import os, sys
import cv2
import torch
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import estimate_transform, warp, resize, rescale

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg


def warp_back(image, oldimage, tform):

    alpha = 0.6

    oldimage = oldimage.astype(np.float64) /255.
    new_size = oldimage.shape

    dst_image = warp(image, tform, output_shape=new_size)

    # Mask of non-black pixels.
    mask = np.where(np.all(dst_image == [0, 0, 0], axis=-1))
    dst_image[mask] = oldimage[mask]

    res = cv2.addWeighted(oldimage, 1 - alpha, dst_image, alpha, 0)
    res = res[:, :, ::-1]

    return res


def main(args):
    savefolder = args.savefolder
    device = args.device
    os.makedirs(savefolder, exist_ok=True)

    # load test images
    testdata = datasets.TestData(args.inputpath, iscrop=args.iscrop, face_detector=args.detector)

    # run DECA
    deca_cfg.model.use_tex = args.useTex
    deca = DECA(config=deca_cfg, device=device)
    # for i in range(len(testdata)):
    for i in tqdm(range(len(testdata))):

        name = testdata[i]['imagename']
        images = testdata[i]['image'].to(device)[None, ...]
        original_images = testdata[i]['original_image'].to(device)[None, ...]

        codedict = deca.encode(images)
        opdict, visdict = deca.decode(codedict)  # tensor

        images = util.tensor2image(images[0])
        original_images = util.tensor2image(original_images[0])
        image = util.tensor2image(visdict['shape_detail_images'][0])

        # plt.imshow(image)
        # plt.show()

        new = warp_back(image, original_images, testdata[i]['tform'])
        plt.imshow(new)
        plt.show()

        break




    #     if args.saveDepth or args.saveKpt or args.saveObj or args.saveMat or args.saveImages:
    #         os.makedirs(os.path.join(savefolder, name), exist_ok=True)
    #     # -- save results
    #     if args.saveObj:
    #         deca.save_obj(os.path.join(savefolder, name, name + '.obj'), opdict)
    #     if args.saveMat:
    #         opdict = util.dict_tensor2npy(opdict)
    #         savemat(os.path.join(savefolder, name, name + '.mat'), opdict)
    #     if args.saveVis:
    #         cv2.imwrite(os.path.join(savefolder, name + '_vis.jpg'), deca.visualize(visdict))
    #     if args.saveImages:
    #         for vis_name in ['inputs', 'rendered_images', 'albedo_images', 'shape_images', 'shape_detail_images']:
    #             if vis_name not in visdict.keys():
    #                 continue
    #             image = util.tensor2image(visdict[vis_name][0])
    #             cv2.imwrite(os.path.join(savefolder, name, name + '_' + vis_name + '.jpg'),
    #                         util.tensor2image(visdict[vis_name][0]))
    # print(f'-- please check the results in {savefolder}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DECA: Detailed Expression Capture and Animation')

    parser.add_argument('-i', '--inputpath', default='TestSamples/SRF', type=str,
                        help='path to the test data, can be image folder, image path, image list, video')
    parser.add_argument('-s', '--savefolder', default='TestSamples/SRF/results', type=str,
                        help='path to the output directory, where results(obj, txt files) will be stored.')
    parser.add_argument('--device', default="cuda" if torch.cuda.is_available() else "cpu", type=str,
                        help='set device, cpu for using cpu')
    # process test images
    parser.add_argument('--iscrop', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to crop input image, set false only when the test image are well cropped')
    parser.add_argument('--detector', default='fan', type=str,
                        help='detector for cropping face, check decalib/detectors.py for details')
    # save
    parser.add_argument('--useTex', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to use FLAME texture model to generate uv texture map, \
                            set it to True only if you downloaded texture model')
    parser.add_argument('--saveVis', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization of output')
    parser.add_argument('--saveKpt', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save 2D and 3D keypoints')
    parser.add_argument('--saveDepth', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save depth image')
    parser.add_argument('--saveObj', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .obj, detail mesh will end with _detail.obj. \
                            Note that saving objs could be slow')
    parser.add_argument('--saveMat', default=False, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save outputs as .mat')
    parser.add_argument('--saveImages', default=True, type=lambda x: x.lower() in ['true', '1'],
                        help='whether to save visualization output as seperate images')
    main(parser.parse_args())