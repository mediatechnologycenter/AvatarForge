# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

import os
import json
import glob
from base_options import Text2SpeachOptions
from googletrans import Translator
from google.cloud import texttospeech


def find_file_ext(data_path, dataset, video_type, name):
    tmp = os.path.join(data_path, dataset, video_type, name + '.txt')

    if os.path.isfile(tmp):
        return tmp

    else:
        print('No text file!')
        exit()

def translate(source_path, target_path, language_source, language_target):
    f = open(source_path, 'r')
    source_txt = f.read()
    f.close()

    if not(language_source==language_target):
        translator = Translator()
        print(f'Source text in {language_source}:\n{source_txt}')
        translation = translator.translate(source_txt,src=language_source, dest=language_target).text
        print(f'Translated text in {language_target}:\n{translation}')

    else:
        translation = source_txt

    f = open(target_path, 'w')
    f.write(translation)
    f.close()

    return

def list_voices(language_code=None):
    client = texttospeech.TextToSpeechClient()
    response = client.list_voices(language_code=language_code)
    voices = sorted(response.voices, key=lambda voice: voice.name)

    print(f" Voices: {len(voices)} ".center(60, "-"))
    for i, voice in enumerate(voices):
        languages = ", ".join(voice.language_codes)
        name = voice.name
        gender = texttospeech.SsmlVoiceGender(voice.ssml_gender).name
        rate = voice.natural_sample_rate_hertz
        print(f"{i} | {languages:<8} | {name:<24} | {gender:<8} | {rate:,} Hz")

    return voices

def text_to_wav(voice_name, text, target_audio_path):
    language_code = "-".join(voice_name.split("-")[:2])
    text_input = texttospeech.SynthesisInput(text=text)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code=language_code, name=voice_name
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16
    )

    client = texttospeech.TextToSpeechClient()
    response = client.synthesize_speech(
        input=text_input, voice=voice_params, audio_config=audio_config
    )

    filename = f"{target_audio_path}"
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to "{filename}"')

# def text2speach(text_path, language):
#     f = open(text_path, 'r')
#     mytext = f.read()
#     f.close()
#
#     myobj = gTTS(text=mytext, lang=language, slow=False)
#     target_audio_path = text_path[:-4]+'.wav'
#     myobj.save(target_audio_path)
#
#     return

def text2speach(text_path, language):
    f = open(text_path, 'r')
    mytext = f.read()
    f.close()

    voices = list_voices(language)
    idx = input('Pick voice index: ')
    voice = voices[int(idx)]

    target_audio_path = text_path[:-4]+'.wav'
    text_to_wav(voice.name, mytext, target_audio_path)

    return

if __name__ == '__main__':

    opt = Text2SpeachOptions().parse()

    data_path = opt.dataroot
    dataset_path = opt.dataset_path
    dataset = opt.dataset
    video_type = opt.video_type
    name = opt.name
    language_source = opt.language_source
    language_target = opt.language_target

    source_path = find_file_ext(data_path, dataset, video_type, name)
    target_path = source_path[:-4]+'_'+language_target+'.txt'

    # Translate input txt to target language
    translate(source_path, target_path, language_source, language_target)

    # Text to speech
    text2speach(target_path, language_target)



