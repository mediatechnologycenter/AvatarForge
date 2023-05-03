
# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
eval "$(conda shell.bash hook)"
conda activate deepspeech

DATAROOT=$1
NAME=$2

TARGET_FPS=$3
SAMPLERATE=$4

TYPE=audio

tracker=--use_DECA

STEPS=(12 0)

for STEP in ${STEPS[@]}
do
  python -W ignore preprocessing.py --dataroot $DATAROOT \
                                    --name $NAME \
                                    --target_fps $TARGET_FPS \
                                    --preprocessing_type $TYPE \
                                    --step $STEP \
                                    $tracker

done

conda deactivate


