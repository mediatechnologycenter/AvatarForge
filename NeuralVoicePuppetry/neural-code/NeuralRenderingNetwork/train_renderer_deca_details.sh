set -ex
# . train_audio2expressionsAttentionTMP.sh &
GPUID=0

DP=/home/alberto/NeuralVoicePuppetry/datasets/External
SUB=Youtube_Russian_guy

DATASETS_DIR=$DP/$SUB
DATASET_MODE=custom_aligned
OBJECT=Youtube_Russian_guy.h5

# neural texture, not used here
TEX_DIM=128
TEX_FEATURES=16

INPUT_NC=5
NUM_THREADS=1

# loss
LOSS=VGG
#LOSS=L1
#LOSS=RMS
#LOSS=L4

# models
TEXTUREMODEL=DynamicNeuralTextureExpression
#TEXTUREMODEL=DynamicNeuralTextureAudio
MODEL=DynamicNeuralTextures
RENDERER_TYPE=UNET_6_level  # There are many more of these TODO


# optimizer parameters
#LR=0.00001
LR=0.0001

#N_ITER=150 #50 #N_ITER=150
#N_ITER_LR_DECAY=50

N_ITER=50 #50 #N_ITER=150
N_ITER_LR_DECAY=50

BATCH_SIZE=8
SEQ_LEN=8


RENDERER=$OBJECT
EROSION=0.6

################################################################################
################################################################################
################################################################################
DATE_WITH_TIME=`date "+%Y%m%d-%H%M%S"`
NAME=$MODEL-$RENDERER_TYPE-$TEXTUREMODEL-SL$SEQ_LEN-BS$BATCH_SIZE-$OBJECT-$DATASET_MODE-$LOSS-$DATE_WITH_TIME-look_ahead_mask_mouth
#NAME=DynamicNeuralTextures-UNET_6_level-DynamicNeuralTextureExpression-SL8-BS8-Halbtotale_355_9415.h5-custom_aligned-VGG-20210308-004445-look_ahead
#NAME=DynamicNeuralTextures-UNET_6_level-DynamicNeuralTextureExpression-SL8-BS8-Halbtotale_355_9415.h5-custom_aligned-VGG-20210309-184531-look_ahead_masked
DISPLAY_NAME=${MODEL}-$DATASET_MODE_${OBJECT}-${RENDERER_TYPE}-SL$SEQ_LEN-BS$BATCH_SIZE-${LOSS}-look_ahead
DISPLAY_ID=0



# training
# --input_noise_augmentation
echo "---------"
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

# # testing
#EPOCH=latest
#python test.py --seq_len $SEQ_LEN --write_no_images --name $NAME --erosionFactor $EROSION --epoch $EPOCH --display_winsize 512 --tex_dim $TEX_DIM --tex_features $TEX_FEATURES --rendererType $RENDERER_TYPE --lossType $LOSS --dataroot $DATASETS_DIR/$OBJECT --model $MODEL --netG unet_256 --dataset_mode $DATASET_MODE --norm instance  --gpu_ids $GPUID

################################################################################
################################################################################
################################################################################
