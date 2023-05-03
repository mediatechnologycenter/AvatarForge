# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import cv2
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys, getopt

def compute_diff(image1, image2):
    # Convert images to grayscale
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between two images
    (score, diff) = structural_similarity(image1, image2, full=True)
    diff = (diff * 255).astype("uint8")

    return score, diff

def main(argv):

    fake_video_path = argv[0]
    real_video_path = argv[1]

    print(f'Fake video path: {fake_video_path}')
    print(f'Real video path: {real_video_path}')

    name = fake_video_path.split('/')[-1][:-4]

    vidcap_fake = cv2.VideoCapture(fake_video_path)
    vidcap_real = cv2.VideoCapture(real_video_path)

    length = min(int(vidcap_fake.get(cv2.CAP_PROP_FRAME_COUNT)), int(vidcap_real.get(cv2.CAP_PROP_FRAME_COUNT)))

    success_fake,image_fake = vidcap_fake.read()
    success_real,image_real = vidcap_real.read()

    best_pair = {
        'index': 0,
        'score': -1,
        'diff': None,
        'real_img': None,
        'fake_img': None,
    }

    for count in tqdm(range(length)):

        score, diff = compute_diff(image_fake, image_real)
        if score > best_pair["score"]:
            best_pair["index"] = count
            best_pair["score"] = score
            best_pair["diff"] = diff
            best_pair["real_img"] = image_real
            best_pair["fake_img"] = image_fake

        success_fake, image_fake = vidcap_fake.read()
        success_real, image_real = vidcap_real.read()

    # best_pair["real_img"] = cv2.cvtColor(best_pair["real_img"], cv2.COLOR_BGR2RGB)
    # best_pair["fake_img"] = cv2.cvtColor(best_pair["fake_img"], cv2.COLOR_BGR2RGB)

    # fig=plt.figure(figsize=(14, 6))
    # fig.suptitle(f'Index: {best_pair["index"]}, Score: {best_pair["score"]}')
    # fig.add_subplot(131)
    # plt.imshow(best_pair["real_img"])
    # fig.add_subplot(132)
    # plt.imshow(best_pair["fake_img"])
    # fig.add_subplot(133)
    # plt.imshow(best_pair["diff"])
    # plt.show()

    os.makedirs('../results/differences/', exist_ok=True)

    cv2.imwrite('results/differences/' + name + '_real_img.png', best_pair["real_img"])
    cv2.imwrite('results/differences/' + name + '_fake_img.png', best_pair["fake_img"])
    cv2.imwrite('results/differences/' + name + '_diff_img.png', best_pair["diff"])


# python find_closest_frames.py '/home/alberto/NeuralVoicePuppetry/results/inference/DynamicNeuralTextures-Demo_Male_moderator_1/Demo_Male_moderator_1_to_Demo_Male_moderator_1/Demo_Male_moderator_1_to_Demo_Male_moderator_1.mp4' '/home/alberto/data/videosynth/External/Demo/Male_moderator_1.mp4'
if __name__ == "__main__":
   main(sys.argv[1:])
