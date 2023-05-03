# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import subprocess
from autils.deca_flame_fitting import *
import cv2
from tqdm import tqdm
import json
import torch


test_sample = '/home/alberto/NeuralVoicePuppetry/third/DECA/TestSamples/SRF/355_9414.jpg'
# Flags to control preprocessing

# Create Tracker object
tracker = DECA_tracker(test_sample)

# extract face tracking information


# Load video
for i in tqdm(range(len(tracker.testdata))):
    images = tracker.testdata[i]['image'].to(tracker.device)[None, ...]
    codedict = tracker(images)
