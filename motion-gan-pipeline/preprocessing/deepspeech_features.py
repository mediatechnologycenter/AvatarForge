"""
20.10.14
Code collection from VOCA:
https://github.com/TimoBolkart/voca
"""
import argparse
import re
import subprocess

from scipy.io import wavfile
import os
import copy
import resampy
import numpy as np
import tensorflow as tf
from python_speech_features import mfcc
import moviepy.editor as mp
from pydub import AudioSegment
from subprocess import call

class AudioHandler:
    def __init__(self, config):
        self.config = config
        self.audio_feature_type = config['audio_feature_type']
        self.num_audio_features = config['num_audio_features']
        self.audio_window_size = config['audio_window_size']
        self.audio_window_stride = config['audio_window_stride']
        self.target_fps = config["target_fps"]

    def process(self, audio):
        if self.audio_feature_type.lower() == "none":
            return None
        elif self.audio_feature_type.lower() == 'deepspeech':
            return self.convert_to_deepspeech(audio)
        else:
            raise NotImplementedError("Audio features not supported")

    def convert_to_deepspeech(self, audio):
        def audioToInputVector(audio, fs, numcep, numcontext):
            # Get mfcc coefficients
            features = mfcc(audio, samplerate=fs, numcep=numcep)

            # We only keep every second feature (BiRNN stride = 2)
            features = features[::2]

            # One stride per time step in the input
            num_strides = len(features)

            # Add empty initial and final contexts
            empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
            features = np.concatenate((empty_context, features, empty_context))

            # Create a view into the array with overlapping strides of size
            # numcontext (past) + 1 (present) + numcontext (future)
            window_size = 2 * numcontext + 1
            train_inputs = np.lib.stride_tricks.as_strided(
                features,
                (num_strides, window_size, numcep),
                (features.strides[0], features.strides[0], features.strides[1]),
                writeable=False)

            # Flatten the second and third dimensions
            train_inputs = np.reshape(train_inputs, [num_strides, -1])

            train_inputs = np.copy(train_inputs)
            train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)

            # Return results
            return train_inputs

        if type(audio) == dict:
            pass
        else:
            raise ValueError('Wrong type for audio')

        # Load graph and place_holders

        with tf.io.gfile.GFile(self.config['deepspeech_graph_fname'], "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        from tensorflow.python.framework.ops import get_default_graph
        graph = get_default_graph()
        tf.import_graph_def(graph_def, name="deepspeech")
        input_tensor = graph.get_tensor_by_name('deepspeech/input_node:0')
        # input_tensor = graph.get_tensor_by_name('input_node:0')
        seq_length = graph.get_tensor_by_name('deepspeech/input_lengths:0')
        layer_6 = graph.get_tensor_by_name('deepspeech/logits:0')

        n_input = 26
        n_context = 9

        processed_audio = copy.deepcopy(audio)
        with tf.compat.v1.Session(graph=graph) as sess:
            for subj in audio.keys():
                for seq in audio[subj].keys():
                    print('process audio: %s - %s' % (subj, seq))

                    audio_sample = audio[subj][seq]['audio']
                    sample_rate = audio[subj][seq]['sample_rate']
                    resampled_audio = resampy.resample(audio_sample.astype(float), sample_rate, 16000)
                    input_vector = audioToInputVector(resampled_audio.astype('int16'), 16000, n_input, n_context)

                    network_output = sess.run(layer_6, feed_dict={input_tensor: input_vector[np.newaxis, ...],
                                                                  seq_length: [input_vector.shape[0]]})

                    # Resample network output from 50 fps to 60 fps
                    audio_len_s = float(audio_sample.shape[0]) / sample_rate
                    num_frames = int(round(audio_len_s * self.target_fps))
                    network_output = self.interpolate_features(network_output[:, 0], 50, self.target_fps,
                                                               output_len=num_frames)

                    # Make windows
                    zero_pad = np.zeros((int(self.audio_window_size / 2), network_output.shape[1]))
                    network_output = np.concatenate((zero_pad, network_output, zero_pad), axis=0)
                    windows = []
                    for window_index in range(0, network_output.shape[0] - self.audio_window_size,
                                              self.audio_window_stride):
                        windows.append(network_output[window_index:window_index + self.audio_window_size])

                    processed_audio[subj][seq]['audio'] = np.array(windows)
        return processed_audio

    def interpolate_features(self, features, input_rate, output_rate, output_len=None):
        num_features = features.shape[1]
        input_len = features.shape[0]
        seq_len = input_len / float(input_rate)
        if output_len is None:
            output_len = int(seq_len * output_rate)
        input_timestamps = np.arange(input_len) / float(input_rate)
        output_timestamps = np.arange(output_len) / float(output_rate)
        output_features = np.zeros((output_len, num_features))
        for feat in range(num_features):
            output_features[:, feat] = np.interp(output_timestamps,
                                                 input_timestamps,
                                                 features[:, feat])
        return output_features


class AudioFeatures:
    def __init__(self, config):
        self.audio_handler = AudioHandler(config)

    def process_audio(self, audio, sample_rate):
        tmp_audio = {'subj': {'seq': {'audio': audio, 'sample_rate': sample_rate}}}
        return self.audio_handler.process(tmp_audio)['subj']['seq']['audio']

    def inference_interpolate_styles(self, audio_fname):
        # TODO: check if this works
        # sample_rate, audio = wavfile.read(audio_fname)
        import soundfile as sf
        import librosa
        import wavio

        x,_ = librosa.load(audio_fname, sr=16000)
        sf.write(audio_fname, x, 16000)
        print('Successfully converted.')

        wav = wavio.read(audio_fname)
        audio, sample_rate = wav.data, wav.rate

        # audio, sample_rate = librosa.load(audio_fname)
        
        print("sample rate: ", sample_rate)
        if audio.ndim != 1:
            print('Audio has multiple channels, only first channel is considered')
            audio = audio[:, 0]

        processed_audio = self.process_audio(audio, sample_rate)
        return processed_audio

    def run(self, audio_fname):
        features = self.inference_interpolate_styles(audio_fname)
        return features


class FFMPEGMetaReader:
    """
    Uses ffprobe to extract metadata from audio and video
    """
    
    def __init__(cls, fps):
        cls.DEFAULTS = {
            "samplerate": 48000,
            "bitrate": 96,
            "duration": "00",
            "fps": fps
        }
    
        cls.PATTERNS = {
            "sample_and_bitrate": r'Audio:.*\s(\d+)\sHz.*\s(\d+)\skb/s',
            "duration": r"Duration:\s(\d{2}):(\d{2}):(\d{2})\.(\d{2})",
            "fps": r"Video:.*\s(\d+\.?\d*)\sfps"
        }

    def _run_ffmpeg(cls, path):
        process = subprocess.Popen(['ffprobe', '-i', path, '-hide_banner'], stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        output = stderr.decode('utf-8')
        return output

    def _sample_and_bitrate(cls, path, default_samplerate=None, default_bitrate=None, output=None):
        if output is None:
            output = cls._run_ffmpeg(path)
        match = re.search(cls.PATTERNS["sample_and_bitrate"], output)

        samplerate = default_samplerate if default_samplerate else cls.DEFAULTS["samplerate"]
        bitrate = default_bitrate if default_bitrate else cls.DEFAULTS["bitrate"]

        if match:
            samplerate = match.group(1) if match.group(1) else samplerate
            bitrate = match.group(2) if match.group(2) else bitrate
        return samplerate, bitrate

    def _duration(cls, path, output=None):
        if output is None:
            output = cls._run_ffmpeg(path)

        duration = cls.DEFAULTS["duration"]

        match = re.search(cls.PATTERNS["duration"], output)

        if match and match.group(4):
            duration = "{}:{}:{}.{}".format(*[match.group(i) if match.group(i) else duration for i in range(1, 5)])
        return duration

    def _fps(cls, video_path, output=None):
        if output is None:
            output = cls._run_ffmpeg(video_path)

        fps = cls.DEFAULTS["fps"]

        match = re.search(cls.PATTERNS["fps"], output)

        if match:
            fps = match.group(1) if match.group(1) else fps
        else:
            raise Warning("No fps found.")
        return fps

    def extract_meta(cls, path):
        output = cls._run_ffmpeg(path)
        samplerate, bitrate = cls._sample_and_bitrate(path, output=output)
        duration = cls._duration(path, output=output)
        fps = cls._fps(path, output=output)
        return {
            "samplerate": samplerate,
            "bitrate": bitrate,
            "duration": duration,
            "fps": fps
        }


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, required=True)
    parser.add_argument("--video_id", type=str, required=True)
    return parser


def get_config():
    config = {}
    config['deepspeech_graph_fname'] = os.path.join(
        "third/DeepSpeech/models/", "output_graph.pb")
    config['audio_feature_type'] = 'deepspeech'
    config['num_audio_features'] = 29

    config['audio_window_size'] = 16
    config['audio_window_stride'] = 1
    # config['target_fps'] = target_fps
    return config


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def extract_ds(dataset_base, name, file_type, sample_rate=16000, target_fps=25.0):

    config = get_config()
    path_wav = '{}/{}.wav'.format(dataset_base, name)

    # First convert mp4 to wav
    if file_type == 'video' and not os.path.isfile(path_wav):

        path_video_in = '{}/{}.mp4'.format(dataset_base, name)

        if not os.path.isfile(path_video_in):
            path_video_in = '{}/{}.avi'.format(dataset_base, name)

        # cmd = "ffmpeg -i input-video.avi -vn -acodec copy output-audio.aac"

        cmd = ('ffmpeg' + f' -i {path_video_in} -vn -acodec copy -ar {sample_rate} {path_wav} ').split()
        call(cmd)

    else:
        if not os.path.isfile(path_wav):
            try:
                path_mp3 = '{}/{}.mp3'.format(dataset_base, name)
                sound = AudioSegment.from_mp3(path_mp3)
                sound.export(path_wav, format="wav")

            except FileNotFoundError:
                print('Audio file not found. File format should be mp3 or wav.')

    config["target_fps"] = target_fps

    # Compute features
    features = AudioFeatures(config=config).run(path_wav)
    folder_out = os.path.join(dataset_base, "audio_feature")

    # Save features
    mkdir(folder_out)
    for i, feature in enumerate(features):
        fn_out = "%05d.deepspeech.npy" % i
        np.save(os.path.join(folder_out, fn_out), feature)
    print("Written {} files to '{}'".format(features.shape[0], folder_out))


if __name__ == "__main__":
    args = get_parser().parse_args()
    folder_videos = args.dataset
    file_id = args.video_id
    folder_nvp = "TARGETS"

    # First convert mp4 to wav
    path_video_in = '{}/{}.mp4'.format(folder_videos, file_id)
    path_wav = '{}/{}.wav'.format(folder_videos, file_id)
    path_features = os.path.join(folder_videos, "deepspeech", "{}.npy".format(file_id))
    clip = mp.VideoFileClip(path_video_in)
    clip.audio.write_audiofile(path_wav)

    # Get the meta data via ffmpeg
    config = get_config()
    metadata = FFMPEGMetaReader.extract_meta(path_video_in)
    if int(metadata["fps"]) != float(metadata["fps"]):
        raise Warning("Careful: fps is not an integer ({})".format(metadata["fps"]))
    config["target_fps"] = int(metadata["fps"])

    # Compute features
    features = AudioFeatures(config=config).run(path_wav)
    folder_out = os.path.join(
        './NeuralVoicePuppetry/Audio2ExpressionNet/Inference/datasets/{}'.format(
            folder_nvp), file_id, "audio_feature")

    # Save features
    mkdir(folder_out)
    for i, feature in enumerate(features):
        fn_out = "%05d.deepspeech.npy" % i
        np.save(os.path.join(folder_out, fn_out), feature)
    print("Written {} files to '{}'".format(features.shape[0], folder_out))
