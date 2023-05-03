# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import mediapipe as mp
import os
from PIL import Image, ImageDraw
import cv2
import numpy as np
from tqdm import tqdm

def extract_mediapipe(input_path, output_path):

    # process video
    frames_path = os.path.join(input_path, 'frames')
    
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) 

    for idx, img_path in enumerate(tqdm([os.path.join(frames_path, f) for f in sorted(os.listdir(frames_path))])):
        image = cv2.imread(img_path)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))   

        save_path = os.path.join(output_path, '%05d.npy' % idx)

        if not results.pose_landmarks:
            ldk_poses = np.zeros(shape=(33, 4)) 

        else:
            ldk_poses = []
            # Save as numpy
            for i in range(33):
                # landmarks: x, y, z, visibility
                ldk = results.pose_landmarks.landmark[i]
                ldk_poses.append([ldk.x * image_width , ldk.y * image_height , ldk.z, ldk.visibility])

        np.save(save_path, ldk_poses)

        # Save landmarks and edges for DEBUGGING
        debug = False
        if debug:
            debug_path = os.path.join(output_path, 'debug')        
            os.makedirs(debug_path, exist_ok=True)

            annotated_image = image.copy()
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = (192, 192, 192) # gray
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            
            cv2.imwrite(os.path.join(debug_path, str(idx) + '.png'), annotated_image)

            tmp = Image.open(img_path)
            size = (image_height, image_width)

            for point in ldk_poses:
                margin = (max(size) // 500) + 3
                ldmks = ([point[0] - margin, point[1] - margin, point[0] + margin, point[1] + margin])
                draw = ImageDraw.Draw(tmp)
                draw.ellipse(ldmks, fill=(255))
                tmp.save(os.path.join(debug_path, str(idx) + '_landmarks.png'))
        

if __name__ == "__main__":

    input_path = '/home/alberto/motion-gan-pipeline/input_data/video/Clara'
    output_path = '/home/alberto/motion-gan-pipeline/input_data/video/Clara/body_pose'
    extract_mediapipe(input_path, output_path)
