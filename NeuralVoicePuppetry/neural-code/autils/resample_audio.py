# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from scipy.io import wavfile

audio_fname = '/home/alberto/data/videosynth/SRF_anchor_short/Halbtotale/355_9415.wav'
new_audio = '/home/alberto/data/videosynth/SRF_anchor_short/Halbtotale/355_9415_short_Clara.wav'

samplerate, data = wavfile.read(audio_fname)

new_data = data[:20*samplerate, :]

wavfile.write(new_audio, samplerate, new_data)
