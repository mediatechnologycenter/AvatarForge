# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

GPUID=0

BASE=`pwd`
echo $BASE
export GOOGLE_APPLICATION_CREDENTIALS="/home/alberto/ttsnvp-a5777a0c2445.json"

# DATA PATHS
DATAROOT=/home/alberto/data/videosynth/
DATASETPATH=/home/alberto/NeuralVoicePuppetry/datasets/

### SOURCE ###
# Italian
DATASET_SOURCE=Synthetic
VIDEOTYPE_SOURCE=Italian
NAME_SOURCE_LIST=( clara )
LANGUAGE_SOURCE='it'
LANGUAGE_TARGET_LIST=( 'it' 'en' 'es' )

## English
#DATASET_SOURCE=Synthetic
#VIDEOTYPE_SOURCE=English
#NAME_SOURCE_LIST=( how_are_you )
#LANGUAGE_SOURCE='en'
#LANGUAGE_TARGET='it'

######################################

### TARGET ###
# Jennifer
DATASET_TARGET=SRF_anchor_short
VIDEOTYPE_TARGET=Halbtotale
NAME_TARGET=355_9415

## Vicky
#DATASET_TARGET=External
#VIDEOTYPE_TARGET=Italian
#NAME_TARGET=Vicky_EAC

TARGET_FPS=25

##################### PREPROCESSING #####################
for NAME_SOURCE in "${NAME_SOURCE_LIST[@]}"
do
  OG_NAME=$NAME_SOURCE
  for LANGUAGE_TARGET in "${LANGUAGE_TARGET_LIST[@]}"
  do
    echo 'Generating audio from source text: '$DATAROOT$DATASET_SOURCE/$VIDEOTYPE_SOURCE/$NAME_SOURCE
    python text_to_speach.py \
    --dataroot $DATAROOT \
    --dataset_path $DATASETPATH \
    --dataset $DATASET_SOURCE \
    --video_type $VIDEOTYPE_SOURCE \
    --name $NAME_SOURCE \
    --language_source $LANGUAGE_SOURCE \
    --language_target $LANGUAGE_TARGET \

    NAME_SOURCE=$OG_NAME'_'$LANGUAGE_TARGET

    echo 'Preprocessing source video: '$DATAROOT$DATASET_SOURCE/$VIDEOTYPE_SOURCE/$NAME_SOURCE
    python preprocessing.py \
    --dataroot $DATAROOT \
    --dataset_path $DATASETPATH \
    --dataset $DATASET_SOURCE \
    --video_type $VIDEOTYPE_SOURCE \
    --name $NAME_SOURCE \
    --preprocess_ds \
    --target_fps $TARGET_FPS \

    echo 'Preprocessing target video: '$DATAROOT$DATASET_TARGET/$VIDEOTYPE_TARGET/$NAME_TARGET
    python preprocessing.py \
    --dataroot $DATAROOT \
    --dataset_path $DATASETPATH \
    --dataset $DATASET_TARGET \
    --video_type $VIDEOTYPE_TARGET \
    --name $NAME_TARGET \
    --preprocess_ds \
    --target_fps $TARGET_FPS \
    --skip_h5py \

    # Twice for memory issues
    python preprocessing.py \
    --dataroot $DATAROOT \
    --dataset_path $DATASETPATH \
    --dataset $DATASET_TARGET \
    --video_type $VIDEOTYPE_TARGET \
    --name $NAME_TARGET \
    --preprocess_tracking \
    --target_fps $TARGET_FPS \

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

    # SOURCE
    SOURCE_ACTOR_LIST=( \
    $DATASETPATH$DATASET_SOURCE/$VIDEOTYPE_SOURCE'_'$NAME_SOURCE \
    )

    # TARGET
    TARGET_ACTOR_LIST=( \
    $DATASETPATH$DATASET_TARGET/$VIDEOTYPE_TARGET'_'$NAME_TARGET \
    )

    rm -f ./datasets/TRANSFERS/$NAME/list_transfer.txt
    cd Audio2ExpressionNet/Inference/
    pwd
    for TARGET_ACTOR in "${TARGET_ACTOR_LIST[@]}"
    do
        echo 'Training for Target: '$TARGET_ACTOR

        for SOURCE_ACTOR in "${SOURCE_ACTOR_LIST[@]}"
        do
            echo 'Training for Source: '$SOURCE_ACTOR
            #  --look_ahead
            python transfer.py --look_ahead --seq_len $SEQ_LEN --source_actor $SOURCE_ACTOR --target_actor $TARGET_ACTOR --write_no_images --name $A2E_NAME --erosionFactor $EROSION --epoch $EPOCH --display_winsize 512 --rendererType $RENDERER_TYPE --lossType $LOSS --dataroot $DATASETS_DIR/$OBJECT --model $MODEL --netG unet_256 --dataset_mode $DATASET_MODE --norm instance  --gpu_ids $GPUID
        done
    done
    cd ..
    cd ..

    ##################### NEURAL RENDERING #####################

    echo -e '\n--------------Neural Rendering Training---------------\n'
    ############ TRAINING ############
    DP=$DATASETPATH$DATASET_TARGET
    SUB=$VIDEOTYPE_TARGET
    DATASETS_DIR=$DP/$VIDEOTYPE_TARGET'_'$NAME_TARGET
    OBJECT=$VIDEOTYPE_TARGET'_'$NAME_TARGET'.h5'

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
  #  TEXTUREMODEL=DynamicNeuralTextureAudio
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
    NAME=$MODEL-$VIDEOTYPE_TARGET'_'$NAME_TARGET
    #NAME=$TEXTUREMODEL-$VIDEOTYPE_TARGET'_'$NAME_TARGET
    #NAME=DynamicNeuralTextures-Halbtotale_355_9415

    DISPLAY_NAME=$MODEL-$VIDEOTYPE_TARGET'_'$NAME_TARGET
    #DISPLAY_NAME=$TEXTUREMODEL-$VIDEOTYPE_TARGET'_'$NAME_TARGET
    DISPLAY_ID=0
    #NAME=DynamicNeuralTextures-UNET_6_level-DynamicNeuralTextureExpression-SL8-BS8-Halbtotale_355_9415.h5-custom_aligned-VGG-20210311-095742-look_ahead_mask_mouth

    cd NeuralRenderingNetwork

    if [ -d "checkpoints/$NAME" ]; then
      echo "Model $NAME already exists."


    else
      echo "Training model $NAME..."

      CUDA_VISIBLE_DEVICES=0 python \
      train_renderer.py \
      --textureModel $TEXTUREMODEL \
      --num_threads $NUM_THREADS \
      --input_nc $INPUT_NC \
      --display_id $DISPLAY_ID \
      --look_ahead \
      --seq_len $SEQ_LEN  \
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
      --pool_size 0  \
      --gpu_ids $GPUID \
      --lr $LR \
      --batch_size $BATCH_SIZE \
      --deca_details \
      #--continue_train \
      #--epoch_count 15 \

    fi

    ############ INFERENCE ############
    echo -e '\n--------------Neural Rendering Inference---------------\n'

    DT=$DATASETPATH'TRANSFERS'
    TRANSFER_PATH=$DT/$A2E_NAME
    CHECKPOINT_DIR=$BASE/NeuralRenderingNetwork/checkpoints

    FILE_ID_SOURCE=$VIDEOTYPE_SOURCE'_'$NAME_SOURCE
    FILE_ID_TARGET=$VIDEOTYPE_TARGET'_'$NAME_TARGET

    DR=$DATASETPATH$DATASET_TARGET/$FILE_ID_TARGET/
    SOURCE_DATAROOT=$DATASETPATH$DATASET_SOURCE/$FILE_ID_SOURCE/$FILE_ID_SOURCE.h5
    TARGET_DATAROOT=$DATASETPATH$DATASET_TARGET/$FILE_ID_TARGET/$FILE_ID_TARGET.h5
    EXPR_PATH=$TRANSFER_PATH/$FILE_ID_SOURCE--$FILE_ID_TARGET/expressions

    IMAGES_TARGET_DIR=$BASE/results/inference/$NAME/$FILE_ID_SOURCE"_to_"$FILE_ID_TARGET

    # Used as start and end
    FRAME_ID_SOURCE=-1
    FRAME_ID_TARGET=-1

    CUDA_VISIBLE_DEVICES=0 python \
    inference_renderer.py \
    --num_threads $NUM_THREADS \
    --input_nc $INPUT_NC  \
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
    --norm instance   \
    --gpu_ids $GPUID \
    --batch_size $BATCH_SIZE \
    --epoch latest \
    --textureModel $TEXTUREMODEL \
    --dataroot $DR \
    --target_dataroot $TARGET_DATAROOT \
    --source_dataroot $SOURCE_DATAROOT \
    --expr_path $EXPR_PATH \
    --images_target_dir $IMAGES_TARGET_DIR \
    --frame_id_source $FRAME_ID_SOURCE \
    --frame_id_target $FRAME_ID_TARGET \
    --deca_details \

    cd ..

    ##################### POSTPROCESSING #####################
    echo -e '\n--------------Postprocessing---------------\n'
    python postprocessing.py \
    --file_id_source $FILE_ID_SOURCE \
    --file_id_target $FILE_ID_TARGET \
    --model_name $NAME \
    --frames_path $DATAROOT$DATASET_TARGET/$VIDEOTYPE_TARGET/$NAME_TARGET \
    --audio_fname $DATAROOT$DATASET_SOURCE/$VIDEOTYPE_SOURCE/$NAME_SOURCE'.wav' \
    --dataset_target $DATASETPATH$DATASET_TARGET \
    --target_fps $TARGET_FPS

  done
done


