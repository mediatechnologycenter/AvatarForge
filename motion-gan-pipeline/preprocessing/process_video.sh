# '''
# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details
# '''
eval "$(conda shell.bash hook)"
conda activate deepspeech
DATAROOT=$1
NAME=$2

TARGET_FPS=$3
SAMPLERATE=$4

TYPE=video

tracker=--use_DECA

# STEPS=(12 0 1 2 3 4 5 9 11 13)
STEPS=(12 0 1 2 3 4 5 9 11)

process=$true

if $process;
then
  # # ffmpeg -i "$entry/$tmp.mp4" -r $TARGET_FPS -y "$entry/$tmp.tmp.mp4"
  # mkvmerge --default-duration 0:${TARGET_FPS}fps --fix-bitstream-timing-information 0 "$DATAROOT/$NAME/$NAME.mp4" -o "$DATAROOT/$NAME/$NAME.tmp.mkv"
  # ffmpeg -i "$DATAROOT/$NAME/$NAME.tmp.mkv" -c:v copy -y "$DATAROOT/$NAME/$NAME.mp4"
  # rm "$DATAROOT/$NAME/$NAME.tmp.mkv"

  ffmpeg -i "$DATAROOT/$NAME/$NAME.mp4" -ar $SAMPLERATE -y "$DATAROOT/$NAME/$NAME.wav"

fi

for STEP in ${STEPS[@]}
do

  if [ $STEP = "0" ]; 
  then
    conda activate deepspeech
  else
    conda activate pyenv
  fi
  python -W ignore preprocessing.py --dataroot $DATAROOT \
                                    --name $NAME \
                                    --target_fps $TARGET_FPS \
                                    --preprocessing_type $TYPE \
                                    --step $STEP \
                                    $tracker
done

conda deactivate