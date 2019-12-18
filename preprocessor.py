import os
import sys
sys.path.append('/home/olga/Projects/tacotron2_waveglow_multispeaker_gst')
sys.path.append('/home/olga/Projects/tacotron2_waveglow_multispeaker_gst/processing')
sys.path.append('/home/olga/Projects/tacotron2_waveglow_multispeaker_gst/waveglow')
import time
import pathlib
import librosa
from tqdm import tqdm
from shutil import copyfile
from scipy.io import wavfile
from multiprocessing import Pool
from tacotron2.text import text_to_sequence
import pandas as pd
import numpy as np
from hparams import PreprocessingConfig
np.random.seed(42)

config=PreprocessingConfig()

SR=config.SR
TOP_DB=config.TOP_DB
limit_by=config.limit_by
minimum_viable_dur=config.minimum_viable_dur
N=config.N
output_directory=config.output_directory
data=config.data

lines=[]
def process(path, output_directory, speaker_name, speaker_id, process_audio=True):
    with open(os.path.join(path, 'metadata.csv'), 'r') as file:
        files_to_process = []
        output_path = os.path.join(output_directory, speaker_name)
        output_audio_path = os.path.join(output_path, 'wavs')
        inter_audio_path = os.path.join(output_path, 'wavs_inter')
        pathlib.Path(output_audio_path).mkdir(parents=True, exist_ok=True)
        #pathlib.Path(inter_audio_path).mkdir(parents=True, exist_ok=True)

        for line in file:
            parts = line.strip().split('|')
            file_name = parts[0]
            text = parts[1]
            if len(parts) == 3:
                text = parts[2]
            if not file_name.endswith('.wav'):
                file_name = file_name + '.wav'

            input_file_path = os.path.join(path, 'wavs', file_name)
            inter_file_path = os.path.join(inter_audio_path, file_name)
            final_file_path = os.path.join(output_audio_path, file_name)
            files_to_process.append((input_file_path, inter_file_path, final_file_path, process_audio, file_name, text, speaker_name))
            new_line = '|'.join([final_file_path, text, str(speaker_id)]) + '\n'
            lines.append(new_line)

        #with open(os.path.join(output_path, 'data.txt'), 'a+') as f:
        #    f.writelines(lines)

    return files_to_process

def mapper(job):
    fin, fint, fout, process_audio, file_name, text, speaker_name= job
    seq = text_to_sequence(text, ['english_cleaners'])
    data, _ = librosa.load(fin, sr=SR)

    if process_audio:
        data, _ = librosa.effects.trim(data, top_db=TOP_DB)
        dur_librosa=librosa.get_duration(data)
        #wavfile.write(fint, SR, data)
        wavfile.write(fin, SR, data)
        command = "ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar {} {} -nostats -loglevel 0".format(fin, SR, fout)
        os.system(command)
    else:
        dur_librosa = librosa.get_duration(data)
        copyfile(fin, fout)

    return fout, len(seq), dur_librosa, speaker_name


def main(output_directory, data):
    """
    Parse commandline arguments.
    data: list of tuples (source_directory, speaker_id, process_audio_flag)
    """
    jobs = []
    for source_directory, speaker_id, process_audio_flag in tqdm(data):
        for path, dirs, files in os.walk(source_directory):
            if 'wavs' in dirs and 'metadata.csv' in files:
                speaker_name = source_directory.split('/')[-1]
                sub_jobs = process(path, output_directory, speaker_name, speaker_id, process_audio_flag)
                jobs += sub_jobs
    print('Files to convert:', len(jobs))
    time.sleep(5)

    with Pool(42) as p:
       results=p.map(mapper, jobs)
    distribution=pd.DataFrame({
        'path': [r[0] for r in results],
        'text': [r[1] for r in results],
        'dur': [r[2] for r in results],
        'speaker': [r[3] for r in results]
    })
    speakers=list(distribution['speaker'].unique())

    if limit_by in speakers:
        #speakers.remove(limit_by)
        limiting_distribution=distribution[distribution['speaker']==limit_by]
        mind=min(limiting_distribution['dur'])
        maxd = max(limiting_distribution['dur'])
        mint = min(limiting_distribution['text'])
        maxt = max(limiting_distribution['text'])
        print('Min {} text: {}'.format(limit_by, mint))
        print('Max {} text: {}'.format(limit_by, maxt))
        print('Min {} dur: {}'.format(limit_by, mind))
        print('Max {} dur: {}'.format(limit_by, maxd))
        print('-----------------------------------------')

    train_lines = []
    val_lines = []
    for speaker in speakers:
        df=distribution[distribution['speaker']==speaker]
        df=df[(df['text']<=maxt) & (df['text']>=1) & (df['dur']<=maxd) & (df['dur']>=minimum_viable_dur)]
        msk = np.random.rand(len(df)) < 0.95
        train = df[msk]
        val = df[~msk]

        if len(train) > N:
            train = train.sample(N, random_state=42)

        train_set = train['path'].unique()
        val_set = val['path'].unique()
        print('Train set for {} is: {}'.format(speaker, len(train)))
        print('Val set for {} is: {}'.format(speaker, len(val)))
        print('-----------------------------------------')

        for line in lines:
            file_path, _, _ = line.strip().split('|')
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
    main(output_directory, data)


