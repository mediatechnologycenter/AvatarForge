#!/bin/bash

# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

set -e
eval "$(conda shell.bash hook)"

# SET GPU DEVICE
GPUID=0

# BASE='/mnt/full_pipeline' 
BASE=$1
echo "DATAROOT: $BASE"

NAME_AUDIO_FILE=$2
NAME_VIDEO_FILE=$3
echo "NAME_AUDIO_FILE: $2"
echo "NAME_VIDEO_FILE: $3"

# You can choose here what you want
FPS=25
SAMPLERATE=16000

# DATA PATHS
INPUTDATAPATH=$BASE/input_data
OUTPUTDATAPATH=$BASE/output_data
FEATURESPATH=features

###AUDIO###
NAME_AUDIOS_LIST=( $NAME_AUDIO_FILE )

### VIDEO ###
NAME_VIDEOS_LIST=( $NAME_VIDEO_FILE )

for NAME_AUDIO in "${NAME_AUDIOS_LIST[@]}"
do
  for NAME_VIDEO in "${NAME_VIDEOS_LIST[@]}"
  do

    # if [ ! -d "$INPUTDATAPATH/audio/$NAME_AUDIO" ]; then
    #   echo "Moving audio file.."
    #   mkdir -p $INPUTDATAPATH/audio/$NAME_AUDIO
    #   mv $INPUTDATAPATH/audio/$NAME_AUDIO.wav $INPUTDATAPATH/audio/$NAME_AUDIO/$NAME_AUDIO.wav 
    # fi

    # if [ ! -d "$INPUTDATAPATH/video/$NAME_VIDEO" ]; then
    #   echo "Moving video file.."
    #   mkdir -p $INPUTDATAPATH/video/$NAME_VIDEO
    #   mv $INPUTDATAPATH/video/$NAME_VIDEO.mp4 $INPUTDATAPATH/video/$NAME_VIDEO/$NAME_VIDEO.mp4 
    # fi

    # Make output folder
    tmp='_to_'
    now="$(date +'_%d%m%y_%X')"
    OUTPUTFOLDER=$OUTPUTDATAPATH/$NAME_AUDIO$tmp$NAME_VIDEO
    # OUTPUTFOLDER=$OUTPUTDATAPATH/$NAME_AUDIO$tmp$NAME_VIDEO$now
    echo $OUTPUTFOLDER
    mkdir -p $OUTPUTFOLDER

    ##################### PREPROCESSING #####################
    echo -e '\n--------------PREPROCESSING---------------\n'
    cd preprocessing
    if [ ! -f "$INPUTDATAPATH/video/$NAME_VIDEO/track_params.pt" ]
    then
    bash process_video.sh $INPUTDATAPATH/video $NAME_VIDEO $FPS $SAMPLERATE
    fi

    bash process_audio.sh $INPUTDATAPATH/audio $NAME_AUDIO $FPS $SAMPLERATE
    cd ..
    conda deactivate

    #################### AUDIO EXPRESSIONS #####################
    echo -e '\n--------------AUDIO EXPRESSIONS---------------\n'
    conda activate pyenv
    cd preprocessing
    python third/Audio2ExpressionNet/get_audioexpr.py --name $NAME_AUDIO --dataset_base $INPUTDATAPATH/audio/$NAME_AUDIO --out_dir $OUTPUTFOLDER --mapping_path $INPUTDATAPATH/video/$NAME_VIDEO/mapping.npy
    cd ..

    ##################### HEAD MOTION #####################
    echo -e '\n--------------HEAD MOTION---------------\n'
    cd motion-generation
    python transfer.py --dataroot $INPUTDATAPATH --dataset_names $NAME_AUDIO --target_name $NAME_VIDEO --out_dir $OUTPUTFOLDER --checkpoint_dir $BASE/checkpoints/
    cd ..

    ##################### HEAD TO SHOULDERS #####################
    echo -e '\n--------------HEAD TO BODY---------------\n'
    python face2body.py --dataset_base $INPUTDATAPATH/video/$NAME_VIDEO --target_name $NAME_VIDEO --checkpoint_dir $BASE/checkpoints/

    #################### EDGE MAP #####################
    echo -e '\n--------------EDGE MAP---------------\n'
    cd preprocessing
    python combine_edges.py --dataset_base $INPUTDATAPATH/video/$NAME_VIDEO --out_dir $OUTPUTFOLDER --target_name $NAME_VIDEO --checkpoint_dir $BASE/checkpoints/
    cd ..
    
    ##################### GAN INFERENCE #####################
    echo -e '\n--------------GAN INFERENCE---------------\n'
    cd ImageToImage
    python generate_images.py --dataroot $INPUTDATAPATH/video --video_name $NAME_VIDEO --input_test_root_dir $OUTPUTFOLDER/edges/ --out_dir $OUTPUTFOLDER/generated_frames/ --checkpoint_dir $BASE/checkpoints/
    cd ..

    ##################### POSTPROCESSING #####################
    echo -e '\n--------------POSTPROCESSING---------------\n'
    python postprocessing.py --dataroot $INPUTDATAPATH --name_audio $NAME_AUDIO --out_dir $OUTPUTFOLDER --fps $FPS --sr $SAMPLERATE --clean --move_to_one_folder

    echo -e ' ================= ALL DONE ================= '
    conda deactivate


  done

done
