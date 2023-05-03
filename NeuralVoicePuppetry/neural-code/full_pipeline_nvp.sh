#!/bin/bash

# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

set -e
eval "$(conda shell.bash hook)"
# cd ../neural-code

GPUID=0
BASE_DIR="${3:-$(pwd)}"
echo "BASE_DIR=$BASE_DIR"

NAME_AUDIO_FILE=$1
NAME_VIDEO_FILE=$2
echo "NAME_AUDIO_FILE: $1"
echo "NAME_VIDEO_FILE: $2"

# DATA PATHS
INPUTDATAPATH=$BASE_DIR/input_data
OUTPUTDATAPATH=$BASE_DIR/output_data
FEATURESPATH=features

#Testing output path
OUTPUT_TOOL_DATA_PATH=$BASE_DIR/output_data/videos
#mkdir -p $OUTPUT_TOOL_DATA_PATH

IMAGES_TARGET_DIR=$OUTPUT_TOOL_DATA_PATH

###AUDIO###
AUDIO_PATH=audio
NAME_AUDIOS_LIST=($NAME_AUDIO_FILE)

### VIDEO ###
VIDEO_PATH=video
NAME_VIDEOS_LIST=($NAME_VIDEO_FILE)

TARGET_FPS=25

##################### PREPROCESSING #####################
for NAME_AUDIO in "${NAME_AUDIOS_LIST[@]}"; do
  for NAME_VIDEO in "${NAME_VIDEOS_LIST[@]}"; do
    # Set time variable
    SECONDS=0

    conda activate deepspeech
    if [ $NAME_AUDIO != $NAME_VIDEO ]; then
      echo 'Preprocessing source audio: '$INPUTDATAPATH/$AUDIO_PATH/$NAME_AUDIO
      python preprocessing.py \
        --dataroot $INPUTDATAPATH \
        --dataset_path $OUTPUTDATAPATH/$FEATURESPATH \
        --dataset $AUDIO_PATH \
        --name $NAME_AUDIO \
        --preprocess_ds \
        --target_fps $TARGET_FPS \
        --clean

    else
      echo $NAME_AUDIO'='$NAME_VIDEO

    fi
  
    echo 'Preprocessing target video: '$INPUTDATAPATH/$VIDEO_PATH/$NAME_VIDEO
    python preprocessing.py \
    --dataroot $INPUTDATAPATH \
    --dataset_path $OUTPUTDATAPATH/$FEATURESPATH \
    --dataset $VIDEO_PATH \
    --name $NAME_VIDEO \
    --preprocess_ds \
    --target_fps $TARGET_FPS \
    --skip_h5py \

    conda deactivate

    # Twice for memory issues
    conda activate pyenv
    python preprocessing.py \
    --dataroot $INPUTDATAPATH \
    --dataset_path $OUTPUTDATAPATH/$FEATURESPATH \
    --dataset $VIDEO_PATH \
    --name $NAME_VIDEO \
    --preprocess_tracking \
    --target_fps $TARGET_FPS \
    --clean \

    ##################### AUDIO2EXPRESSION #####################
    echo -e '\n--------------AUDIO2EXPRESSION---------------\n'
    OBJECT=ARD_ZDF
    LR=0.00001
    N_ITER=150
    N_ITER_LR_DECAY=50
    RENDERER=$OBJECT
    EROSION=1.0
    BATCH_SIZE=16
    MODEL=audio2ExpressionsAttentionTMP4
    RENDERER_TYPE=estimatorAttention
    DATASET_MODE=multi_face_audio_eq_tmp_cached
    LOSS=RMS
    SEQ_LEN=8
    DATE_WITH_TIME=20191105-115332
    A2E_NAME=$MODEL-$RENDERER_TYPE-SL$SEQ_LEN-BS$BATCH_SIZE-$OBJECT-$DATASET_MODE-$LOSS-$DATE_WITH_TIME-look_ahead
    EPOCH=latest

    DT=$OUTPUTDATAPATH/'TRANSFERS'
    TRANSFER_PATH=$DT/$A2E_NAME

    # AUDIO
    AUDIO_LIST=($OUTPUTDATAPATH/$FEATURESPATH/$NAME_AUDIO)

    # VIDEO
    VIDEO_LIST=($OUTPUTDATAPATH/$FEATURESPATH/$NAME_VIDEO)

    rm -f $OUTPUTDATAPATH/TRANSFERS/$NAME/list_transfer.txt
    cd Audio2ExpressionNet/Inference/
    pwd
    for TARGET_ACTOR in "${VIDEO_LIST[@]}"; do
      echo 'Training for Target: '$TARGET_ACTOR

      for SOURCE_ACTOR in "${AUDIO_LIST[@]}"; do
        echo 'Training for Source: '$SOURCE_ACTOR
        #  --look_ahead
        python transfer.py --look_ahead --base_path $OUTPUTDATAPATH --seq_len $SEQ_LEN --source_actor $SOURCE_ACTOR --target_actor $TARGET_ACTOR --write_no_images --name $A2E_NAME --erosionFactor $EROSION --epoch $EPOCH --display_winsize 512 --rendererType $RENDERER_TYPE --lossType $LOSS --dataroot $DATASETS_DIR/$OBJECT --model $MODEL --netG unet_256 --dataset_mode $DATASET_MODE --norm instance --gpu_ids $GPUID --transfer_path $TRANSFER_PATH
      done
    done
    cd ..
    cd ..

    ##################### NEURAL RENDERING #####################

    echo -e '\n--------------Neural Rendering Training---------------\n'
    ############ TRAINING ############
    DP=$OUTPUTDATAPATH/$FEATURESPATH
    DATASETS_DIR=$DP/$NAME_VIDEO
    OBJECT=$NAME_VIDEO'.h5'

    ############# CHECKPOINTS PATH ############
    CHECKPOINT_DIR=$OUTPUTDATAPATH/checkpoints

    #dataset mode
    DATASET_MODE=custom_aligned
    # neural texture, not used here
    TEX_DIM=128
    TEX_FEATURES=16
    INPUT_NC=5
    NUM_THREADS=1
    # loss
    LOSS=VGG
    # models
    TEXTUREMODEL=DynamicNeuralTextureExpression
    # TEXTUREMODEL=DynamicNeuralTextureAudio
    MODEL=DynamicNeuralTextures
    RENDERER_TYPE=UNET_6_level
    # optimizer parameters
    LR=0.0001
    N_ITER=50
    N_ITER_LR_DECAY=50
    BATCH_SIZE=8
    SEQ_LEN=8
    RENDERER=$OBJECT
    EROSION=0.6

    # Model name
    NAME=$MODEL-$NAME_VIDEO

    DISPLAY_NAME=$MODEL-$NAME_VIDEO
    DISPLAY_ID=0

    cd NeuralRenderingNetwork

    if [ -f "$CHECKPOINT_DIR/$NAME/latest_texture.pth" ]; then
      echo "Model $NAME already exists."

    else
      echo "Training model $NAME..."

      CUDA_VISIBLE_DEVICES=0 python \
        train_renderer.py \
        --textureModel $TEXTUREMODEL \
        --num_threads $NUM_THREADS \
        --input_nc $INPUT_NC \
        --checkpoints_dir $CHECKPOINT_DIR \
        --display_id $DISPLAY_ID \
        --look_ahead \
        --seq_len $SEQ_LEN \
        --save_latest_freq 100000 \
        --no_augmentation \
        --name $NAME \
        --erosionFactor $EROSION \
        --tex_dim $TEX_DIM \
        --tex_features $TEX_FEATURES \
        --rendererType $RENDERER_TYPE \
        --lossType $LOSS \
        --display_env $DISPLAY_NAME \
        --niter $N_ITER \
        --niter_decay $N_ITER_LR_DECAY \
        --dataroot $DATASETS_DIR/$OBJECT \
        --model $MODEL \
        --netG unet_256 \
        --lambda_L1 100 \
        --dataset_mode $DATASET_MODE \
        --no_lsgan \
        --norm instance \
        --pool_size 0 \
        --gpu_ids $GPUID \
        --lr $LR \
        --batch_size $BATCH_SIZE \
        --deca_details
      #--continue_train \
      #--epoch_count 40 \

    fi

    ############ INFERENCE ############
    echo -e '\n--------------Neural Rendering Inference---------------\n'

    FILE_ID_SOURCE=$NAME_AUDIO
    FILE_ID_TARGET=$NAME_VIDEO

    DR=$OUTPUTDATAPATH/$FEATURESPATH/$FILE_ID_TARGET/
    SOURCE_DATAROOT=$OUTPUTDATAPATH/$FEATURESPATH/$FILE_ID_SOURCE/$FILE_ID_SOURCE.h5
    TARGET_DATAROOT=$OUTPUTDATAPATH/$FEATURESPATH/$FILE_ID_TARGET/$FILE_ID_TARGET.h5
    EXPR_PATH=$TRANSFER_PATH/$FILE_ID_SOURCE--$FILE_ID_TARGET/expressions

    # Used as start and end
    FRAME_ID_SOURCE=-1
    FRAME_ID_TARGET=-1

    CUDA_VISIBLE_DEVICES=0 python \
      inference_renderer.py \
      --num_threads $NUM_THREADS \
      --input_nc $INPUT_NC \
      --look_ahead \
      --seq_len $SEQ_LEN \
      --no_augmentation \
      --name $NAME \
      --checkpoints_dir $CHECKPOINT_DIR \
      --erosionFactor $EROSION \
      --tex_dim $TEX_DIM \
      --tex_features $TEX_FEATURES \
      --rendererType $RENDERER_TYPE \
      --lossType $LOSS \
      --model $MODEL \
      --netG unet_256 \
      --dataset_mode $DATASET_MODE \
      --norm instance \
      --gpu_ids $GPUID \
      --batch_size $BATCH_SIZE \
      --epoch latest \
      --textureModel $TEXTUREMODEL \
      --dataroot $DR \
      --target_dataroot $TARGET_DATAROOT \
      --source_dataroot $SOURCE_DATAROOT \
      --expr_path $EXPR_PATH \
      --images_target_dir $OUTPUTDATAPATH/$FEATURESPATH \
      --frame_id_source $FRAME_ID_SOURCE \
      --frame_id_target $FRAME_ID_TARGET \
      --deca_details

    #--images_target_dir $IMAGES_TARGET_DIR \
    cd ..

    ##################### POSTPROCESSING #####################
    echo -e '\n--------------Postprocessing---------------\n'
    python postprocessing.py \
      --file_id_source $FILE_ID_SOURCE \
      --file_id_target $FILE_ID_TARGET \
      --model_name $NAME \
      --frames_path $OUTPUTDATAPATH/$FEATURESPATH/$NAME_VIDEO/'og_frames/' \
      --audio_fname $INPUTDATAPATH/$AUDIO_PATH/$NAME_AUDIO/$NAME_AUDIO'.wav' \
      --dataset_target $OUTPUTDATAPATH/$FEATURESPATH \
      --target_fps $TARGET_FPS \
      --results_out_dir $IMAGES_TARGET_DIR \
      --clean 

  done

done

conda deactivate
