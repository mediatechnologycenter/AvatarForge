GPUID=0

BASE=`pwd`
echo $BASE

# DATA PATHS
DATAROOT=/home/alberto/data/videosynth/
DATASETPATH=/home/alberto/NeuralVoicePuppetry/datasets/

### SOURCE ###

# Severin Videos
DATASET_SOURCE=External
VIDEOTYPE_SOURCE=Severin_videos
NAME_SOURCE_LIST=( transformers_lecture )

TARGET_FPS=25
##################### PREPROCESSING #####################
for NAME_SOURCE in "${NAME_SOURCE_LIST[@]}"
do
  echo 'Preprocessing source video: '$DATAROOT$DATASET_SOURCE/$VIDEOTYPE_SOURCE/$NAME_SOURCE
  python preprocessing.py \
  --dataroot $DATAROOT \
  --dataset_path $DATASETPATH \
  --dataset $DATASET_SOURCE \
  --video_type $VIDEOTYPE_SOURCE \
  --name $NAME_SOURCE \
  --preprocess_ds \
  --target_fps $TARGET_FPS \
  --clean \


  ##################### AUDIO2EXPRESSION #####################
  echo -e '\n--------------AUDIO2EXPRESSION---------------\n'
  OBJECT=ARD_ZDF
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
  SOURCE_ACTOR=$DATASETPATH$DATASET_SOURCE/$VIDEOTYPE_SOURCE'_'$NAME_SOURCE
  MAPPING_PATH=/home/alberto/NeuralVoicePuppetry/mappings/audio2ExpressionsAttentionTMP4-estimatorAttention-SL8-BS16-ARD_ZDF-multi_face_audio_eq_tmp_cached-RMS-20191105-115332-look_ahead/mapping_Severin_videos_SC
  OUTDIR=/home/alberto/data/dave_fxfy/audio2exprNVP/

  cd Audio2ExpressionNet/Inference/
  python audio2expr_with_map.py --use_mapping --mapping_path $MAPPING_PATH --out_dir $OUTDIR --look_ahead --seq_len $SEQ_LEN --source_actor $SOURCE_ACTOR --write_no_images --name $A2E_NAME --erosionFactor $EROSION --epoch $EPOCH --display_winsize 512 --rendererType $RENDERER_TYPE --lossType $LOSS --dataroot $DATASETS_DIR/$OBJECT --model $MODEL --netG unet_256 --dataset_mode $DATASET_MODE --norm instance  --gpu_ids $GPUID
done
