# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'third/')))

import cv2
import numpy as np
from typing import ClassVar, Dict
import matplotlib.pyplot as plt
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from densepose import add_densepose_config
from densepose.vis.base import CompoundVisualizer
from densepose.vis.bounding_box import ScoredBoundingBoxVisualizer
from densepose.vis.extractor import CompoundExtractor, create_extractor
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from pathlib import Path


def get_extractor_and_visualizer():

    VISUALIZERS: ClassVar[Dict[str, object]] = {
    "dp_contour": DensePoseResultsContourVisualizer,
    "dp_segm": DensePoseResultsFineSegmentationVisualizer,
    "dp_u": DensePoseResultsUVisualizer,
    "dp_v": DensePoseResultsVVisualizer,
    "bbox": ScoredBoundingBoxVisualizer,
    }

    vis_specs = ['dp_contour', 'bbox']
    visualizers = []
    extractors = []
    for vis_spec in vis_specs:
        vis = VISUALIZERS[vis_spec]()
        visualizers.append(vis)
        extractor = create_extractor(vis)
        extractors.append(extractor)
    visualizer = CompoundVisualizer(visualizers)
    extractor = CompoundExtractor(extractors)

    context = {
        "extractor": extractor,
        "visualizer": visualizer
    }

    visualizer = context["visualizer"]
    extractor = context["extractor"]

    return extractor, visualizer


def predict(img, cfg):
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)['instances']
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = np.tile(image[:, :, np.newaxis], [1, 1, 3])
    
    extractor, visualizer = get_extractor_and_visualizer()

    data = extractor(outputs)
    image_vis = visualizer.visualize(image, data)
    return image_vis, data


def extract_denseposes(video_name, input_path, output_path):

    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file("third/densepose/densepose_rcnn_R_50_FPN_s1x.yaml")
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.MODEL.WEIGHTS = "third/densepose/model_final_162be9.pkl"

    # process video
    video_path = Path(input_path, video_name + '.mp4')
    captura = cv2.VideoCapture(str(video_path))
    i = 0
    while(captura.isOpened()):

        ret, img = captura.read()
        if ret == False:
            break

        image_vis, data = predict(img, cfg)

        if data[0][0] is None: 
            print("error")
            #cv2.imwrite("image.png", image_vis)
            continue

        uv_map = data[0][0][0].uv*255.0
        uv_map = uv_map.cpu().numpy().transpose(1,2,0)
        labels = data[0][0][0].labels.unsqueeze(2).cpu().numpy()

        npy_file = str(output_path) + '/uv_%5d.npy' % i
        np.save(npy_file, uv_map)
        i +=1

    captura.release()
    #cv2.destroyAllWindows()


if __name__ == "__main__":

    video_name = 'shortTED.mp4'
    input_path = '../input_data/videos'
    densepose_output_path = Path('../output_data', video_name, 'densepose')
    extract_denseposes(video_name, input_path, densepose_output_path)

