import os
import time
import shutil
import pathlib
import librosa
import argparse
import importlib
from shutil import copyfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

# Parse args
parser = argparse.ArgumentParser(description='Pre-processing')
parser.add_argument('--exp', type=str, default=None, required=True, help='Name of an experiment for configs setting.')
args = parser.parse_args()

# Prepare config
# shutil.copyfile(os.path.join('configs', 'experiments', args.exp + '.py'), os.path.join('configs', '__init__.py'))

# Reload Config
configs = importlib.import_module('configs')
configs = importlib.reload(configs)
Config = configs.PreprocessingConfig

# Config dependent imports
from tacotron2.text import text_to_sequence


np.random.seed(42)

emotions_dict = {
    'neutral_normal': 0,
    'calm_normal': 1,
    'calm_strong': 2,
    'happy_normal': 3,
    'happy_strong': 4,
    'sad_normal': 5,
    'sad_strong': 6,
    'angry_normal': 7,
    'angry_strong': 8,
    'fearful_normal': 9,
    'fearful_strong': 10,
    'disgust_normal': 11,
    'disgust_strong': 12,
    'surprised_normal': 13,
    'surprised_strong': 14
    }


def process(speaker_path, output_directory, speaker_name, speaker_id, process_audio=True, emotion_flag=False):
    """
    Parses 'metadata.csv'.
    Args:
        speaker_path: path to the folder with raw data and file 'metadata.csv'
        output_directory: path to where the processed data should be stored
        speaker_name: e.g. 'linda_johnson'
        speaker_id: e.d. 1
        process_audio: flag where to trim and change format using ffmpeg comand

    Returns:
        files_to_process: list of tuples to be processed by mapper
        new_lines: list with lines to form train and validation datasets

    """
    with open(os.path.join(speaker_path, 'metadata.csv'), 'r') as f:
        files_to_process = []
        new_lines = []
        output_path = os.path.join(output_directory, speaker_name)
        output_audio_path = os.path.join(output_path, 'wavs')
        pathlib.Path(output_audio_path).mkdir(parents=True, exist_ok=True)
        emotion = "neutral_normal"

        for line in f:
            parts = line.strip().split('|')
            file_name = parts[0]
            text = parts[1]
            if len(parts) == 3:
                if emotion_flag:
                    text = parts[1]
                    emotion = parts[2]
                else:
                    text = parts[2]

            if not file_name.endswith('.wav'):
                file_name = file_name + '.wav'

            input_file_path = os.path.join(speaker_path, 'wavs', file_name)
            final_file_path = os.path.join(output_audio_path, file_name)
            files_to_process.append((input_file_path, final_file_path, process_audio,
                                     file_name, text, speaker_name, emotion))
            new_line = '|'.join([final_file_path, text, str(speaker_id), str(emotions_dict[emotion])]) + '\n'
            new_lines.append(new_line)

    return files_to_process, new_lines


def mapper(job):

    """
    Measures duration of audio and length of text.
    If speaker_data['process_audio'] == True, trims audio file and  changes its format using ffmpeg.

    Args:
        job: list of tuples (
             path to the audio file,
            path where to put the processed audio file,
            flag do or do not trimming and ffmpeg command,
            name of audio file,
            text description of audio,
            name of speaker e.g. 'linda_johnson')

    Returns: list of tuples (
    path to processed audio file,
    length of text description of audiofile,
    duration of audio,
    name of speaker)
    """
    fin, fout, process_audio, file_name, text, speaker_name, emotion = job
    seq = text_to_sequence(text, ['english_cleaners'])
    data, _ = librosa.load(fin, sr=Config.sr)

    if process_audio:
        data, _ = librosa.effects.trim(data, top_db=Config.top_db)
        dur_librosa = librosa.get_duration(data)
        wavfile.write(fout, Config.sr, data)
        command = "ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar {} {} -nostats -loglevel 0".format(fout, Config.sr, fout)
        os.system(command)
    else:
        dur_librosa = librosa.get_duration(data)
        copyfile(fin, fout)

    return fout, len(seq), dur_librosa, speaker_name, emotion


def main(output_directory, data):
    """
    Args:
        output_directory: path to folder with processed data for all speakers
        data: list of dictionaries with speaker's data (comes from Config)
    """
    lines = []
    if Config.load_data_and_distributions:
        with open(os.path.join(Config.output_directory, 'data.txt'), 'r') as f:
            for line in f:
                lines.append(line)
        distribution = pd.read_csv(os.path.join(Config.output_directory, 'distribution.csv'))
        print('Loaded data.txt and distribution.csv')

    else:
        jobs = []
        for speaker_data in tqdm(data):
            for path, dirs, files in os.walk(speaker_data['path']):
                if 'wavs' in dirs and 'metadata.csv' in files:
                    speaker_name = speaker_data['path'].split('/')[-1]
                    sub_jobs, sub_lines = process(
                        path, output_directory, speaker_name,
                        speaker_data['speaker_id'], speaker_data['process_audio'], speaker_data['emotion'])
                    jobs += sub_jobs
                    for ln in sub_lines:
                        lines.append(ln)
        print('Files to convert:', len(jobs))
        time.sleep(5)

        with Pool(Config.cpus) as p:
            results = p.map(mapper, jobs)

        distribution = pd.DataFrame({
            'path': [r[0] for r in results],
            'text': [r[1] for r in results],
            'dur': [r[2] for r in results],
            'speaker': [r[3] for r in results],
            'emotion': [r[4] for r in results]
        })

        if Config.save_distribution:
            distribution.to_csv(os.path.join(Config.output_directory, "distribution.csv"))

        if Config.save_data_txt:
            with open(os.path.join(Config.output_directory, 'data.txt'), 'a+') as f:
                f.writelines(lines)

    speakers = list(distribution['speaker'].unique())
    maxt = Config.text_limit
    maxd = Config.dur_limit

    if Config.limit_by in speakers:
        limiting_distribution = distribution[distribution['speaker'] == Config.limit_by]
        mind = min(limiting_distribution['dur'])
        maxd = max(limiting_distribution['dur'])
        mint = min(limiting_distribution['text'])
        maxt = max(limiting_distribution['text'])
        print('Min {} text: {}'.format(Config.limit_by, mint))
        print('Max {} text: {}'.format(Config.limit_by, maxt))
        print('Min {} dur: {}'.format(Config.limit_by, mind))
        print('Max {} dur: {}'.format(Config.limit_by, maxd))
        print('-----------------------------------------')

    train_lines = []
    val_lines = []
    for speaker in speakers:
        df = distribution[distribution['speaker'] == speaker]
        df = df[(df['text'] <= maxt) & (df['text'] >= 1) &
                (df['dur'] <= maxd) & (df['dur'] >= Config.minimum_viable_dur)]
        msk = np.random.rand(len(df)) < 0.95
        train = df[msk]
        val = df[~msk]

        if len(train) > Config.n:
            train = train.sample(Config.n, random_state=42)

        train_set = train['path'].unique()
        val_set = val['path'].unique()
        print('Train set for {} is: {}'.format(speaker, len(train)))
        print('Val set for {} is: {}'.format(speaker, len(val)))
        print('-----------------------------------------')

        for line in lines:
            file_path, _, _, _ = line.strip().split('|')
            if (file_path in train_set) & (line not in train_lines):
                train_lines.append(line)
            if (file_path in val_set) & (line not in val_lines):
                val_lines.append(line)

    with open(os.path.join(output_directory, 'train.txt'), 'w') as f:
        f.writelines(train_lines)
    with open(os.path.join(output_directory, 'val.txt'), 'w') as f:
        f.writelines(val_lines)
    print('Done!')


if __name__ == '__main__':
    main(Config.output_directory, Config.data)
