# Video Preprocessing

This is a Python repository for video processing. \
This collection of techniques can be used to process any video file and extract numerous useful features. The preprocessing is project independent and can be used for any Computer Vision application where video processing is required.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```
## Available Processing Techniques
Each processing step can be accessed independently given its reference numbered:

0. <b/>Deepspeech Features </b> \
Extract [Deepspeech](https://arxiv.org/abs/1412.5567) features from the original video audio signal.

1. <b/>Frames Extraction </b> \
Save all frames of the input video.

2. <b/>Landmark Detection </b> \
Extract [FAN](https://github.com/1adrianb/face-alignment) 2D facial landmarks from all video's frames.

3. <b/>Head Pose Estimation </b> \
Run [Deca](https://github.com/YadiraF/DECA) tracker on all frames to extract [FLAME](https://flame.is.tue.mpg.de) morphable model parameters. Then fit a <b/>Rotation</b> matrix and <b/>Translation</b> vector to express face position at each frame. 

4. <b/>Audio Expressions </b> \
Generate DECA facial expressions from Deepspeech audio features.

5. <b/>Background Matting </b> \
Run [RVM](https://github.com/PeterL1n/RobustVideoMatting) to remove the background from each frame.

6. <b/>Extract Meshes </b> \
Create FLAME meshes for each frame of the video.

7. <b/>Save Parameters </b> \
Create JSON file with all transformation parameters. Used for NeRF training.

8. <b/>Speech To Text </b> \
Create text file with transcript of the audio file. 

9. <b/>Body Tracking </b> \
Use [Mediapipe](https://google.github.io/mediapipe/solutions/pose.html) for body pose tracking.

10. <b/>Emotion Detection </b> \
Run [EMOCA](https://github.com/radekd91/emoca) for per-frame emotion detection.

11. <b/>Edge Detection </b> \
Create edge images from FAN facial landmarks and mediapipe body landmarks. Used as input for generative models (GAN).

12. <b/>Noise Reduction </b> \
Clean the audio file removing noise.

13. <b/>Optical Flow </b> \
Use [FlowNet](https://github.com/NVIDIA/flownet2-pytorch) to extract optical flow between consecutive frames.

## Usage

You can run each processing step by using the following command:
```bash
python preprocessing.py --dataroot $DATAROOT 
                        --name $NAME 
                        --target_fps $TARGET_FPS 
                        --preprocessing_type $TYPE 
                        --step $STEP 
                        --use_deca
```

Where:
- ```$DATAROOT```: path to data folder.
- ```$NAME```: name of the video.
- ```$TARGET_FPS```: video's frame per second (suggested: 25).
- ```$TYPE```: preprocessing type: 'audio' or 'video'.
- ```$STEP```: preprocessing step to run.

The data should be organised in the following structure:

```bash 
Data_folder
├── audio # Folder containing n audio files
│   ├── audio_1
│   │   └──audio_1.wav
│   ...
│   └── audio_n
│       └──audio_n.wav
└── video # Folder containing m video files
    ├── video_1
    │   └──video_1.mp4
    ...
    └── video_m
        └──video_m.mp4

```
<b/>Example: </b> \
To run step 0 of preprocessing on <b/>video_1</b> you should run the following command:
```bash
python preprocessing.py --dataroot Data_folder/video/ 
                        --name video_1
                        --target_fps 25
                        --preprocessing_type video
                        --step 0
                        --use_deca
```

## Audio-Driven Video Synthesis Usage

In order to run the necessary preprocessing for the Audio-Driven Video Synthesis project, the following commands are used:

Preprocessing for the input video:
```bash 
bash process_video.sh $INPUTDATAPATH/video $NAME_VIDEO $FPS $SAMPLERATE
```
Preprocessing for the input audio:
```bash 
bash process_audio.sh $INPUTDATAPATH/audio $NAME_AUDIO $FPS $SAMPLERATE
```

Where:
- ```$INPUTDATAPATH```: path to input data folder.
- ```$NAME_VIDEO```: name of the video.
- ```$NAME_AUDIO```: name of the audio.
- ```$FPS```: video's frame per second (suggested: 25).
- ```$SAMPLERATE```: audio's sample rate.

This call in automatically executed in the complete pipeline script.


## Output Folder Structure
By running the previous commands, the following data structure is produced.

```bash 
Video_Name
├── audio_expr # Contains Audio Expressions: .npy
├── audio_feature # Contains Deepspeech Features: .npy
├── body_pose # Contains Mediapipe Landmarks: .npy
├── cropped # Contained cropped frames for training: .png
├── debug # Debug folder
│   ├── debug_mixed # Overlay images, original + FLAME render: .jpg
│   ├── debug_render # FLAME render: .jpg
│   ├── opticalflow # Optical flow visualisation: .png
│   ├── proj_landmarks # 2D projected FAN landmarks: .lms
│   └── proj_landmarks_img # Visualization of projected landmarks: .png
├── deca_expr # DECA expressions: .npy
├── edges # Edge Images: .png
├── expr_masks # DECA mask: .jpg
├── frames # Extracted frames: .jpg
├── landmarks # FAN landmarks: .lms
├── matting # Frames without background: .png
├── opticalflow # Optical flows: .flo
├── Video_Name.mp4 # Original video file
├── Video_Name.wav # Extracted audio file
├── mapping.npy # Learned mapping deepspeech -> FLAME Expressions
└── track_params.pt # Fitted Pose information
```

## License
[MIT](https://choosealicense.com/licenses/mit/)