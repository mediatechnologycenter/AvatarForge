# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

# DATAROOT=/media/apennino/TED384-v2
# DATAROOT=/media/apennino/MTC
# DATASETS=("Test")
DATAROOT=$1
DATASETS=$2

# TARGET_FPS=25
# SAMPLERATE=16000
TARGET_FPS=$3
SAMPLERATE=$4

TYPE=audio

tracker=--use_DECA

STEPS=(0)

for DATASET in ${DATASETS[@]}
do
  for entry in "$DATAROOT/$DATASET"/*
  do
    for STEP in ${STEPS[@]}
    do

      python preprocessing.py --dataroot $DATAROOT/$DATASET \
                        --name $entry \
                        --target_fps $TARGET_FPS \
                        --preprocessing_type $TYPE \
                        --step $STEP \
                        $tracker

    done
  done
done


