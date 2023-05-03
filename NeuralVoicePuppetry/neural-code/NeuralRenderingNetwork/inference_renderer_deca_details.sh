set -ex

DP=/home/alberto/NeuralVoicePuppetry/datasets/SRF_anchor_short
DPS=/home/alberto/NeuralVoicePuppetry/datasets/External
DT=/home/alberto/NeuralVoicePuppetry/datasets/TRANSFERS/
TRANSFER_PATH=$DT/audio2ExpressionsAttentionTMP4-estimatorAttention-SL8-BS16-ARD_ZDF-multi_face_audio_eq_tmp_cached-RMS-20191105-115332-look_ahead
CHECKPOINT_DIR=/home/alberto/NeuralVoicePuppetry/NeuralRenderingNetwork/checkpoints

#FILE_ID_SOURCE=Halbtotale_355_9414
#FILE_ID_SOURCE=Italian_vicky_audio

FILE_ID_SOURCE_LIST=(Alberto_videos_uno Alberto_videos_due Alberto_videos_tre)
#FILE_ID_SOURCE_LIST=(Clara_audios_two)
#FILE_ID_TARGET=Halbtotale_355_9415
FILE_ID_TARGET=Youtube_Russian_guy

# . train_audio2expressionsAttentionTMP.sh &
#EROSION=0.2
# Name for standard model
NAME=DynamicNeuralTextures-UNET_6_level-DynamicNeuralTextureExpression-SL8-BS8-Halbtotale_355_9415.h5-custom_aligned-VGG-20210302-144030-look_ahead
# Name for deca details model
NAME=DynamicNeuralTextures-UNET_6_level-DynamicNeuralTextureExpression-SL8-BS8-Halbtotale_355_9415.h5-custom_aligned-VGG-20210308-004445-look_ahead
# Name for deca details masked model
NAME=DynamicNeuralTextures-UNET_6_level-DynamicNeuralTextureExpression-SL8-BS8-Halbtotale_355_9415.h5-custom_aligned-VGG-20210309-184531-look_ahead_masked
# Name for deca details of masked mouth SRF moderator
#NAME=DynamicNeuralTextures-UNET_6_level-DynamicNeuralTextureExpression-SL8-BS8-Halbtotale_355_9415.h5-custom_aligned-VGG-20210311-095742-look_ahead_mask_mouth
# Name for deca details of masked mouth Russian guy
#NAME=DynamicNeuralTextures-UNET_6_level-DynamicNeuralTextureExpression-SL8-BS8-Youtube_Russian_guy.h5-custom_aligned-VGG-20210315-231511-look_ahead_mask_mouth

EROSION=0.6

GPUID=0
DATASET_MODE=custom_aligned

for FILE_ID_SOURCE in "${FILE_ID_SOURCE_LIST[@]}"
do
  DATAROOT=$DPS/$FILE_ID_TARGET/
  TARGET_DATAROOT=$DPS/$FILE_ID_TARGET/$FILE_ID_TARGET.h5
  #SOURCE_DATAROOT=$DP/$FILE_ID_SOURCE/$FILE_ID_SOURCE.h5
  SOURCE_DATAROOT=$DPS/$FILE_ID_SOURCE/$FILE_ID_SOURCE.h5
  EXPR_PATH=$TRANSFER_PATH/$FILE_ID_SOURCE--$FILE_ID_TARGET/expression.txt

  IMAGES_TARGET_DIR=/home/alberto/NeuralVoicePuppetry/results/inference/$NAME/$FILE_ID_SOURCE"_to_"$FILE_ID_TARGET

  # neural texture, not used here
  TEX_DIM=128
  TEX_FEATURES=16
  TEXTUREMODEL=DynamicNeuralTextureExpression
  INPUT_NC=5
  NUM_THREADS=1

  LOSS=VGG

  # models
  MODEL=DynamicNeuralTextures
  RENDERER_TYPE=UNET_6_level  # There are many more of these TODO

  BATCH_SIZE=1
  SEQ_LEN=8

  ################################################################################
  ################################################################################
  ################################################################################


  #DISPLAY_NAME=DynamicNeuralTextures-UNET_6_level-SL8-BS16-201125_TV-20190114-2245-5701_24fps_cropped_short.h5-custom_aligned-RMS-20201123-185215-look_ahead
  DISPLAY_ID=0

  # Used as start and end
  FRAME_ID_SOURCE=-1
  FRAME_ID_TARGET=-1


  # training
  # --input_noise_augmentation
  echo "---------"

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
  --dataroot $DATAROOT \
  --target_dataroot $TARGET_DATAROOT \
  --source_dataroot $SOURCE_DATAROOT \
  --expr_path $EXPR_PATH \
  --images_target_dir $IMAGES_TARGET_DIR \
  --frame_id_source $FRAME_ID_SOURCE \
  --frame_id_target $FRAME_ID_TARGET \
  --deca_details

done

################################################################################
################################################################################
################################################################################
