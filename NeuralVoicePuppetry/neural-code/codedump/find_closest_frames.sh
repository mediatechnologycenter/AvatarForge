
DATASET_SOURCE=Original_NVP
VIDEOTYPE_SOURCE=Videos
NAME_SOURCE_LIST=( mpi_franziska_neutral_eng obama sequence002 )

for NAME in "${NAME_SOURCE_LIST[@]}"
do
  VIDEONAME=$VIDEOTYPE_SOURCE'_'$NAME
  python find_closest_frames.py '/home/alberto/NeuralVoicePuppetry/results/inference/DynamicNeuralTextures-'$VIDEONAME/$VIDEONAME'_to_'$VIDEONAME/$VIDEONAME'_to_'$VIDEONAME'.mp4' '/home/alberto/data/videosynth/'$DATASET_SOURCE/$VIDEOTYPE_SOURCE/$NAME'.mp4'
done