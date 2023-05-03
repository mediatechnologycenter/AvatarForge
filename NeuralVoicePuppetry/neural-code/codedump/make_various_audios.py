# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from subprocess import call

name = "transformers_lecture"
file_path = "/home/alberto/data/videosynth/External/Severin_videos/"

durations = [ 10 ]

for dur in durations:
    cmd = (f"ffmpeg -i {file_path}{name}.wav -ss 0 -to {dur} -c copy {file_path}{name}_{dur}.wav").split()
    call(cmd)
