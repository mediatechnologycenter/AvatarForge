# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

GPUID=0

BASE=`pwd`
# DATA PATHS
DATAROOT=/home/alberto/data/videosynth/
DATASETPATH=/home/alberto/NeuralVoicePuppetry/datasets/

# SOURCE
DATASET_SOURCE=External
VIDEOTYPE_SOURCE=Youtube
#NAME_SOURCE=Russian_guy
NAME_SOURCE=Bill_Maher

##################### PREPROCESSING #####################

echo 'Preprocessing video: '$DATAROOT$DATASET_SOURCE/$VIDEOTYPE_SOURCE/$NAME_SOURCE
python preprocessing.py \
--dataroot $DATAROOT \
--dataset_path $DATASETPATH \
--dataset $DATASET_SOURCE \
--video_type $VIDEOTYPE_SOURCE \
--name $NAME_SOURCE \
