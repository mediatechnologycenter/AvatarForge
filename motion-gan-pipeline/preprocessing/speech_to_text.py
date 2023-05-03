# SPDX-License-Identifier: MIT
# © 2020-2022 ETH Zurich and other contributors, see AUTHORS.txt for details

from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
import soundfile as sf
import librosa
import os
import torch


def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch


def speech_to_text(name, dataset_base, textpath):
    data, samplerate = librosa.load(os.path.join(dataset_base, name +'.wav'), sr=16000)
    data = torch.from_numpy(data)

    model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-small-librispeech-asr")
    processor = Speech2TextProcessor.from_pretrained("facebook/s2t-small-librispeech-asr")

    inputs = processor(data, sampling_rate=samplerate, return_tensors="pt")
    generated_ids = model.generate(inputs["input_features"], attention_mask=inputs["attention_mask"])

    transcription = processor.batch_decode(generated_ids)

    textfile = open(textpath, "w")
    for element in transcription:
        textfile.write(element + "\n")
    textfile.close()


