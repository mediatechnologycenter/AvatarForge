GPUID=0

BASE=`pwd`
# DATA PATHS
DATAROOT=/home/alberto/data/videosynth/
DATASETPATH=/home/alberto/NeuralVoicePuppetry/datasets/

# SOURCE
DATASET_SOURCE=SRF_anchor_short
VIDEOTYPE_SOURCE=Halbtotale
NAME_SOURCE=355_9415

# TARGET
DATASET_TARGET=SRF_anchor_short
VIDEOTYPE_TARGET=Close
NAME_TARGET=355_9105

#############################################
SOURCE_NAME=$VIDEOTYPE_SOURCE'_'$NAME_SOURCE
TARGET_NAME=$VIDEOTYPE_TARGET'_'$NAME_TARGET
MODEL='TRANSFERS/audio2ExpressionsAttentionTMP4-estimatorAttention-SL8-BS16-ARD_ZDF-multi_face_audio_eq_tmp_cached-RMS-20191105-115332-look_ahead'

AUDIO_PATH = $DATAROOT$DATASET_SOURCE/$VIDEOTYPE_SOURCE/$NAME_SOURCE'.wav'
EXPRESSION_PATH = $DATASETPATH$MODEL/$SOURCE_NAME'--'$TARGET_NAME/'expressions'
CODEDICTS_PATH = $DATASETPATH$DATASET_TARGET/$TARGET_NAME/'DECA_codedicts'
TFORM_PATH = $DATASETPATH$DATASET_TARGET/$TARGET_NAME/'tform.npy'
FRAMES_PATH = $DATAROOT$DATASET_TARGET/$VIDEOTYPE_TARGET/$NAME_TARGET

echo 'Reconstructing face: '$SOURCE_NAME'--'$TARGET_NAME
python reconstruct_faces.py \
--source_name $SOURCE_NAME \
--target_name $TARGET_NAME
--audio_path $AUDIO_PATH \
--expression_path $EXPRESSION_PATH \
--codedicts_path $CODEDICTS_PATH \
--tform_path $TFORM_PATH \
--frames_path $FRAMES_PATH \
