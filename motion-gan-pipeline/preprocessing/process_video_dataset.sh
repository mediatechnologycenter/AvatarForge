# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

#!/bin/bash

eval "$(conda shell.bash hook)"

# DATAROOT=/media/apennino/TED384-v2
# DATAROOT=/media/apennino/MTC
# DATAROOT=/media/apennino/EmotionDetection
# DATAROOT=/media/apennino/TMP_AVSpeech
# DATASETS=("Test")
# DATAROOT=/media/apennino/
# DATASETS=("AI+Art")
DATAROOT=$1
DATASETS=$2

# TARGET_FPS=25
# SAMPLERATE=16000
TARGET_FPS=$3
SAMPLERATE=$4

TYPE=video
#

tracker=--use_DECA

STEPS=(0 2 3 4 5 1 11)
# STEPS=(0 2 3 4 1)
process=$true

for DATASET in ${DATASETS[@]}
do
  for entry in "$DATAROOT/$DATASET"/*
  do
    if $process;
    then
      tmp=$(basename "$entry")
      # ffmpeg -i "$entry/$tmp.mp4" -r $TARGET_FPS -y "$entry/$tmp.tmp.mp4"
      mkvmerge --default-duration 0:${TARGET_FPS}fps --fix-bitstream-timing-information 0 "$entry/$tmp.mp4" -o "$entry/$tmp.tmp.mkv"
      ffmpeg -i "$entry/$tmp.tmp.mkv" -c:v copy -y "$entry/$tmp.mp4"
      rm "$entry/$tmp.tmp.mkv"

      ffmpeg -i "$entry/$tmp.mp4" -ar $SAMPLERATE -y "$entry/$tmp.wav"

    fi

    tmp=$(basename "$entry")

    for STEP in ${STEPS[@]}
    do

      if [ $STEP = "10" ]; 
      then
        conda activate work36_cu11
      else
        conda activate adnerf
      fi
      
      python preprocessing.py --dataroot $DATAROOT/$DATASET \
                        --name $tmp \
                        --target_fps $TARGET_FPS \
                        --preprocessing_type $TYPE \
                        --step $STEP \
                        $tracker

    done
  done
done
