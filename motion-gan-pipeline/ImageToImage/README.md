# Image2Image Generation

This repository contains and implementation of an Image-to-Image translation network used in the Audio-driven Video Synthesis project. \

## Requirements

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements.

```bash
pip install -r requirements.txt
```

## W&B monitoring

All training monitoring can be done on <b/> Weights & Biases </b>. \
You can follow here a detailed guide for [W&B](https://docs.wandb.ai/?_gl=1*1r4dwbe*_ga*OTk1MTkyMDI1LjE2NjQ4MDkyNDY.*_ga_JH1SJHJQXJ*MTY2NDg4MzM1Mi4yLjEuMTY2NDg4MzM1Ny41NS4wLjA.) to set up your own tracking environment.

To run our code you should follow these simple instructions: 
1. Sign up for a free account at https://wandb.ai/site and then login to your wandb account.

2. Install the CLI and Python library for interacting with the Weights and Biases API: 
    ```bash 
    pip install wandb
    ```

3. Log in your wandb account:
    ```bash 
    wandb login
    ```
4. You can now run the code and monitor the experiments on your W&B account.


## Losses

We provide training scripts to experiment with a variety of different training losses:
- L1 loss
- Perceptual VGG loss
- GAN loss
- Optical Flow warping loss for time consistency
- PatchGAN3D Video Discriminator loss for time consistency

All losses to be used can be chosen in the config file. \
During our experiment we found the best combination of loss functions is a mixture of: L1 + VGG Perceptual + PatchGAN3D.

## Train
We provide 3 different scripts for network training:

1. GAN U-Net: \
The first proposed model is a classical Image to Image network. You can regulate training parameters such as losses used or losses weights by making your own config in the <i/>config </i> folder. \
You can run the training using the following command: \
    ```bash
    python train.py -c $CONFIGPATH
                    --checkpoints_dir $CHECKPOINTPATH
                    --input_train_root_dir $INPUTPATH
                    --output_train_root_dir $OUTPUTPATH
                    --height $HEIGHT
                    --width $WIDTH
    ```
    Where:
    - $CONFIGPATH: path to config file (config.train.yml is given).
    - $CHECKPOINTPATH: path to checkpoints directory.
    - $INPUTPATH: path to input images folder.
    - $OUTPUTPATH: path to target images folder.
    - $HEIGHT: height of output.
    - $WIDTH: width of output.
<br/><br/>

2. GAN + Optical Flow \
In addition to the first model, you can train the Image-to-Image network usign optical flow warping loss. \
You can run the training using the following command: 
    ```bash
    python train_optical.py -c $CONFIGPATH
                            --checkpoints_dir $CHECKPOINTPATH
                            --input_train_root_dir $INPUTPATH
                            --output_train_root_dir $OUTPUTPATH
                            --height $HEIGHT
                            --width $WIDTH
    ```
<br/><br/>

3. GAN + PatchGAN3D \
Our final model can be trained using a PatchGAN3D video discriminator. This model is shown to be the best performant in terms of time consistency. \
You can run the training using the following command: 
    ```bash
    python train_temporal.py -c $CONFIGPATH
                             --checkpoints_dir $CHECKPOINTPATH
                             --input_train_root_dir $INPUTPATH
                             --output_train_root_dir $OUTPUTPATH
                             --height $HEIGHT
                             --width $WIDTH
    ```
    
## Inference
You can run inferency by running the following command: \
```bash
python generate_images.py --dataroot $DATAROOT 
                          --video_name $NAME_VIDEO 
                          --input_test_root_dir $INPUTFOLDER
                          --out_dir $OUTPUTFOLDER
                          --checkpoint_dir $CHECKPOINTDIR
```
Where:
- $INPUTDATAPATH: path to the data directory.
- $NAME_VIDEO: name of the target video.
- $INPUTFOLDER: path to input images for the network.
- $OUTPUTFOLDER: path to output folder.
- $CHECKPOINTDIR: path to the existing checkpoint folder.

If no checkoints are found this script will autimatically start training the GAN + PatchGAN3D model.