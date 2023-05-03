# NeuralVoicePuppetry

This repository contains the  end-to-end "Neural Voice Puppetry" for the Audio-driven Video Synthesis project.

You can find additional information on the method in the following [paper](https://arxiv.org/abs/1912.05566).

![NeuralVoicePuppetry](media/pipeline.png "NeuralVoicePuppetry")

## Requirements
Create a new environment:

```bash
# Deepspeech environment
conda create -n deepspeech python=3.7
conda activate deepspeech
conda install pysoundfile -c conda-forge
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html 
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install -r requirements_deepspeech.txt
conda deactivate

# pyenv environment
conda create -n pyenv python=3.9
conda activate pyenv
conda install pysoundfile -c conda-forge 
conda install pytorch=1.13.0 torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c fvcore -c iopath -c conda-forge fvcore iopath 
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
conda install pytorch3d -c pytorch3d
pip install -r requirements.txt
```

## Data
If you plan on using this code with the already available and pre-trained moderators, you will only have to provide the audio data. 

Please follow these instructions on data quality: 
- Audio 
    - Provide a recording of a person speaking (audio of any duration is accepted).
    - The cleaner the audio signal the better: audio with background noise will result in unmatching lip-sync.
    - Avoid recording multiple people talking: the model is unable to distinguish between multiple voice signals.
- Video
    - Provide a video of your desired avatar character talking.
    - Minimum video duration: 3 minutes.
    - Longer videos will results in longer training time.
    - The background is irrelevant.
  
## Data structure

Place your input data using the following structure:

```
input_data
│
└───video
│      └─── NAME_VIDEO_FILE
│      │    └─ NAME_VIDEO_FILE.mp4
│      |...
│
└───audio
│      └─── NAME_AUDIO_FILE
│      |    └─ NAME_AUDIO_FILE.mp3/wav/mp4
│      |...
│   
```

Video files can only be mp4. Moreover for the time being input videos must have 25 fps.

Input audio files can have the following formats: mp4, mp3 or wav.

## Usage
The same command can be used for both training and inference. If the trained checkpoints for the requested avatar are not found, the code will launch the training procedure.

Use the following call to launch the code:
```bash
bash full_pipeline_nvp.sh $AUDIO_NAME $VIDEO_NAME $DATAROOT 
```
Where:
- $AUDIO_NAME: name of the audio file (without extension).
- $VIDEO_NAME: name of the video file (without extension). This will also be the name of the Avatar. 
- $DATAROOT: path to the data folder.

## Train new Avatars
In order to train a new avatar follow this easy steps:

1. Record a new video following the instructions above. \
    Eg: VIDEONAME = my_new_avatar
2. Move your file to your data folder inside the video subfolder. 

3. Choose any audio file. \
    Eg: AUDIONAME = my_audio_sample

4. Start training (and inference) with the following command:
    ```bash
    bash full_pipeline_nvp.sh my_audio_sample my_new_avatar $DATAROOT
    ```

5. Wait until it's finished (if training, the process will be long).
6. Enjoy your new Avatar! \
    You can now use the same avatar with any other audio file.

## Run using pretrained model
In order to run the model on a pretrained (video) identity, you need the following data:
- `./NeuralRenderingNetwork/checkpoints/DynamicNeuralTextures-NAME_VIDEO_FILE`  - this should contain the pretrained model
- `./output_data/features/NAME_VIDEO_FILE` - this should contain the DECA codedics, the original video frames, an h5 file and the tform.npy 

## Computational time 
Estimated computational time : 
* Preprocessing: approx 30 minutes for a 5 minutes video. Varies depending on the frame size.
* Audio2Expression Training: approx 5 minutes for a 5 minutes video.
* Audio2Expression Inference: approx 30 seconds for a 5 minutes video.
* Neural Rendering Training: apporx 8 hours for a 5 minutes video. Change number of epochs for longer videos to keep training time constant.
* Neural Rendering Inference: approx 15 minutes for a 5 minutes video.
* Postprocessing: approx 20 minutes for a 5 minutes video.

## LICENSE

This implementation NVP Pipeline is free and open source! All code in this repository is licensed under:

* [MIT](LICENSE) License.

This pipeline relies and is inspired by the following works, and therefore refers to their individual licenses:

- [Neural Voice Puppetry](https://web.archive.org/web/20201113014501/https://github.com/JustusThies/NeuralVoicePuppetry): [license](https://gitlab.ethz.ch/mtc/video-synthesis/.NeuralVoicePuppetry/-/blob/0fc75ba2edfdfd5655984f1515bfe06de880d91d/LICENSE), internal path `/neural-code/Audio2ExpressionNet`, `/neural-code/NeuralRenderingNetwork`.
- [DECA](https://github.com/YadiraF/DECA): [license](https://github.com/YadiraF/DECA/blob/master/LICENSE), internal path `/neural-code/third/DECA`.
- [VOCA](https://github.com/TimoBolkart/voca): [license](https://voca.is.tue.mpg.de/license), used internally by DECA.
- [DeepSpeech](https://github.com/mozilla/DeepSpeech): [license](https://github.com/mozilla/DeepSpeech/blob/master/LICENSE), used by VOCA.
- [CycleGAN and Pix2Pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix): [license](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/LICENSE), used as skeleton for GAN training.


## Contact information

Alberto Pennino: [alberto.pennino@inf.ethz.ch](alberto.pennino@inf.ethz.ch)


## Reproduce env from scratch

1. Install [gcc](https://linuxize.com/post/how-to-install-gcc-compiler-on-ubuntu-18-04/)

2. Disable [Nouveau kernel driver](https://linuxconfig.org/how-to-disable-nouveau-nvidia-driver-on-ubuntu-18-04-bionic-beaver-linux/) - always reboot 

3. Install [CUDA 11.1](https://developer.nvidia.com/cuda-11.1.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)

Run the following:

```
$ wget https://developer.download.nvidia.com/compute/cuda/11.1.0/local_installers/cuda_11.1.0_455.23.05_linux.run
$ sudo sh cuda_11.1.0_455.23.05_linux.run 
$ export PATH="/usr/local/cuda-11.1/bin:$PATH" export
$ LD_LIBRARY_PATH="/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH"
```

4. Clone repo: git clone https://gitlab.ethz.ch/mtc/NeuralVoicePuppetry.git

5. Install [Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html)

6. Reproduce env

```
conda create -n NAME python=3.8
conda activate NAME
conda install numpy
conda install -c anaconda scipy
conda install -c conda-forge opencv
conda install -c conda-forge tqdm
conda install -c conda-forge resampy
conda install scikit-image
conda install h5py
pip install --upgrade tensorflow
pip install python_speech_features
conda install -c conda-forge moviepy
conda install -c conda-forge pydub
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge librosa
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
pip install face-alignment
conda install -c conda-forge dominate
pip install chumpy
conda install -c conda-forge progressbar2
pip install flask
chmod a+x full_pipeline_nvp.sh
pip install gunicorn
```

<!-- ## Run the api with Gunicorn server

Activate the created environment <br>
`conda activate <name-created-environment>` <br>
Run the server <br>
`gunicorn -w 2 --bind :5000 --error-logfile error.log --timeout 1200 api:app --capture-output --log-level debug` <br>
If your are currently developing you can add the flag for instant reload on code save <br>
`--reload` <br>
If the server is already running, the port 5000 is blocked. Locate it and kill the process to restart it <br>
find the process `ps aux|grep unicorn` and kill it `pkill gunicorn` <br>
Restart again -->



