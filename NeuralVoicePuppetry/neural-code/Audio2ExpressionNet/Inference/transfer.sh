set -ex
# . transfer.sh &
GPUID=0

######################################################
##################   SPECIFY MODEL  ##################
######################################################

## audio2ExpressionsAttentionTMP4-estimatorAttention-SL8-BS16-ARD_ZDF-multi_face_audio_eq_tmp_cached-RMS-20191105-115332-look_ahead # <<<<<<<<<<<<<
DATASETS_DIR=/home/alberto/NeuralVoicePuppetry/datasets
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
NAME=$MODEL-$RENDERER_TYPE-SL$SEQ_LEN-BS$BATCH_SIZE-$OBJECT-$DATASET_MODE-$LOSS-$DATE_WITH_TIME-look_ahead

# --look_ahead
EPOCH=latest

###############################################################
######################   SPECIFY TARGET  ######################
###############################################################

# target actors
#SOURCE_ACTOR_LIST=(/home/alberto/NeuralVoicePuppetry/datasets/External/Alberto_videos_uno /home/alberto/NeuralVoicePuppetry/datasets/External/Alberto_videos_due /home/alberto/NeuralVoicePuppetry/datasets/External/Alberto_videos_tre)
SOURCE_ACTOR_LIST=(/home/alberto/NeuralVoicePuppetry/datasets/External/Clara_audios_one /home/alberto/NeuralVoicePuppetry/datasets/External/Clara_audios_two /home/alberto/NeuralVoicePuppetry/datasets/External/Clara_audios_three)

#TARGET_ACTOR_LIST[1]=/home/alberto/NeuralVoicePuppetry/datasets/External/Youtube_Russian_guy
TARGET_ACTOR_LIST[1]=/home/alberto/NeuralVoicePuppetry/datasets/SRF_anchor_short/Halbtotale_355_9415


rm -f ./datasets/TRANSFERS/$NAME/list_transfer.txt
for TARGET_ACTOR in "${TARGET_ACTOR_LIST[@]}"
do
    echo $TARGET_ACTOR

    for SOURCE_ACTOR in "${SOURCE_ACTOR_LIST[@]}"
    do
        echo $SOURCE_ACTOR
        #  --look_ahead
        python transfer.py --look_ahead --seq_len $SEQ_LEN --source_actor $SOURCE_ACTOR --target_actor $TARGET_ACTOR --write_no_images --name $NAME --erosionFactor $EROSION --epoch $EPOCH --display_winsize 512 --rendererType $RENDERER_TYPE --lossType $LOSS --dataroot $DATASETS_DIR/$OBJECT --model $MODEL --netG unet_256 --dataset_mode $DATASET_MODE --norm instance  --gpu_ids $GPUID
    done
done


###############################################################
###############################################################
###############################################################