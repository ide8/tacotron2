import os
import re
import time
import shutil
import pathlib
import librosa
import argparse
import importlib
from shutil import copyfile
from multiprocessing import Pool
import json

import numpy as np
import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm

# Parse args
parser = argparse.ArgumentParser(description='Pre-processing')
parser.add_argument('--exp', type=str, default=None, required=True, help='Name of an experiment for configs setting.')
args = parser.parse_args()

# Prepare config
shutil.copyfile(os.path.join('configs', 'experiments', args.exp + '.py'), os.path.join('configs', '__init__.py'))

# Reload Config
configs = importlib.import_module('configs')
configs = importlib.reload(configs)
Config = configs.PreprocessingConfig

# Config dependent imports
from tacotron2.text import text_to_sequence

np.random.seed(42)


def process(speaker_path, speaker_name, speaker_id, process_audio=True, emotion_present=False):
    """
    Parses 'metadata.csv'.
    Args:
        speaker_path: path to the folder with raw data and file 'metadata.csv'
        speaker_name: e.g. 'linda_johnson'
        speaker_id: e.d. 1
        process_audio: flag where to trim and change format using ffmpeg comand
        emotion_present: flag which indicates whether emotion is present in speaker data
    Returns:
        jobs: list of tuples to be processed by mapper
    """
    with open(os.path.join(speaker_path, 'metadata.csv'), 'r') as f:
        jobs = []
        output_path = os.path.join(Config.output_directory, speaker_name)
        output_audio_path = os.path.join(output_path, 'wavs')
        pathlib.Path(output_audio_path).mkdir(parents=True, exist_ok=True)
        emotion = 'neutral-normal'

        for line in f:
            parts = line.strip().split('|')
            file_name = parts[0]
            text = parts[1]
            if len(parts) == 3:
                if emotion_present:
                    text = parts[1]
                    emotion = parts[2]
                else:
                    text = parts[2]

            if not file_name.endswith('.wav'):
                file_name = file_name + '.wav'

            input_file_path = os.path.join(speaker_path, 'wavs', file_name)
            final_file_path = os.path.join(output_audio_path, file_name)

            jobs.append((input_file_path, final_file_path, text, speaker_name, speaker_id, emotion, process_audio))

    return jobs


def mapper(job):
    """
    Measures duration of audio and length of text.
    If speaker_data['process_audio'] == True, trims audio file and changes its format using ffmpeg.

    Args:
        job: list of tuples (
            path to the audio file,
            path where to put the processed audio file,
            text transcription of audio,
            name of speaker e.g. 'linda_johnson'
            speaker id e.g. 1,
            emotion e.g. 'neutral-normal',
            process audio flag do or do not trimming and ffmpeg commands
        )
    Returns: list of tuples (
        full path to processed audio file,
        text,
        name of speaker,
        speaker id,
        emotion,
        length of text transcription of audio file,
        duration of audio
    )
    """
    fin, fout, text, speaker_name, speaker_id, emotion, process_audio = job
    seq = text_to_sequence(text, ['english_cleaners'])
    data, _ = librosa.load(fin, sr=Config.sr)

    if process_audio:
        data, _ = librosa.effects.trim(data, top_db=Config.top_db)
        duration = librosa.get_duration(data)

        match = re.match('(.*)(.wav)', fout)
        fint = f'{match.group(1)}-temp{match.group(2)}'

        wavfile.write(fint, Config.sr, data)

        command = 'ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar {} {} -nostats -loglevel 0'.format(fint, Config.sr, fout)
        os.system(command)
        os.remove(fint)
    else:
        duration = librosa.get_duration(data)
        copyfile(fin, fout)

    return fout, text, speaker_name, speaker_id, emotion, len(seq), duration


def balance_coefs(distribution, key):
    """
    :param distribution: pd.DataFrame with file data.csv
    :param key:
    :return: dictionary with keys - dist[key].values, values - coefficients for loss balancing
    """
    true_balance = pd.DataFrame(distribution[key].value_counts() / len(distribution))
    balance = pd.DataFrame({'true_balance': true_balance[key],
                            'sqrt_balance': np.sqrt(true_balance[key])})

    sum_sqrts = sum(balance['sqrt_balance'])
    balance['sqrt_div_sum_sqrts'] = balance['sqrt_balance'] / sum_sqrts
    balance['root'] = np.sqrt(np.divide(sum_sqrts, balance['sqrt_div_sum_sqrts']))
    sum_roots = sum(balance['root'])
    balance['final_balance'] = balance['root'] / sum_roots

    return balance['final_balance'].to_dict()


def main():
    """
    Loads metadata.csv and audio files from wavs directory or restores from data.csv
    Saves data.csv, train.txt, val.txt
    and coefficients_emotions.json, coefficients_emotions.json for loss balancing
    """
    if Config.start_from_preprocessed:
        distribution = pd.read_csv(os.path.join(Config.output_directory, 'data.csv'), sep='|')
        print('Loaded data.csv')
    else:
        jobs = []
        for speaker_data in tqdm(Config.data):
            for speaker_path, dirs, files in os.walk(speaker_data['path']):
                if 'wavs' in dirs and 'metadata.csv' in files:
                    speaker_name = speaker_data['path'].split('/')[-1]
                    speaker_id = speaker_data['speaker_id']
                    process_audio = speaker_data['process_audio']
                    emotion_present = speaker_data['emotion_present']

                    sub_jobs = process(speaker_path, speaker_name, speaker_id, process_audio, emotion_present)
                    jobs += sub_jobs

        print('Files to convert:', len(jobs))
        time.sleep(5)

        with Pool(Config.cpus) as p:
            results = p.map(mapper, jobs)

        distribution = pd.DataFrame({
            'path': [r[0] for r in results],
            'text': [r[1] for r in results],
            'speaker_name': [r[2] for r in results],
            'speaker_id': [r[3] for r in results],
            'emotion': [r[4] for r in results],
            'text_len': [r[5] for r in results],
            'duration': [r[6] for r in results]
        })

        distribution['emotion_id'] = distribution['emotion'].map(Config.emo_id_map)

        distribution.to_csv(os.path.join(Config.output_directory, 'data.csv'), sep='|', index=False)
        print('Saved to data.csv')

    speakers = set(distribution['speaker_name'].unique())
    maxt = Config.text_limit
    maxd = Config.dur_limit

    if Config.limit_by in speakers:
        limiting_distribution = distribution[distribution['speaker_name'] == Config.limit_by]
        mind = min(limiting_distribution['duration'])
        maxd = max(limiting_distribution['duration'])
        mint = min(limiting_distribution['text_len'])
        maxt = max(limiting_distribution['text_len'])

        print('Min {} text: {}'.format(Config.limit_by, mint))
        print('Max {} text: {}'.format(Config.limit_by, maxt))
        print('Min {} dur: {}'.format(Config.limit_by, mind))
        print('Max {} dur: {}'.format(Config.limit_by, maxd))
        print('----------------------------------------------')

    trains, vals = [], []

    for speaker_data in Config.data:
        speaker_name = speaker_data['path'].split('/')[-1]

        df = distribution[distribution['speaker_name'] == speaker_name]
        df = df[(df['text_len'] <= maxt) & (df['text_len'] >= 1) &
                (df['duration'] <= maxd) & (df['duration'] >= Config.minimum_viable_dur)]

        df = df[['path', 'text', 'speaker_id', 'emotion_id']]

        msk = (np.random.rand(len(df)) < 0.95)
        train = df[msk]
        val = df[~msk]

        if len(train) > Config.n:
            train = train.sample(Config.n, random_state=42)

        print('Train set for {} is: {}'.format(speaker_name, len(train)))
        print('Val set for {} is: {}'.format(speaker_name, len(val)))
        print('----------------------------------------------')

        trains.append(train)
        vals.append(val)

    train = pd.concat(trains)
    val = pd.concat(vals)

    e_coefs = balance_coefs(train, 'emotion_id')
    with open(os.path.join(Config.output_directory, 'emotion_coefficients.json'), 'w') as json_file:
        json.dump(e_coefs, json_file)

    s_coefs = balance_coefs(train, 'speaker_id')
    with open(os.path.join(Config.output_directory, 'speaker_coefficients.json'), 'w') as json_file:
        json.dump(s_coefs, json_file)

    train.to_csv(os.path.join(Config.output_directory, 'train.txt'), sep='|', index=False, header=False)
    val.to_csv(os.path.join(Config.output_directory, 'val.txt'), sep='|', index=False, header=False)

    print('Done!')


if __name__ == '__main__':
    main()
