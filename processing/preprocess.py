import os
import sys
import time
import pickle
import librosa
import pathlib
import audioread
from tqdm import tqdm
from shutil import copyfile
from scipy.io import wavfile
from multiprocessing import Pool

from data_pipeline import CODE_PATH, DATA_CONFIG, OUTPUT_DIRECTORY, SR, TOP_DB
from tacotron2.text import text_to_sequence

sys.path.append(CODE_PATH)


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
        p.map(cleaning_mapper, jobs)

    print('Preprocessing done!')
    print('Starting distribution check!')

    dist_data = {}
    for speaker in tqdm(data['data']):
        speaker_name = speaker['name']
        dist_data[speaker_name] = []

        print(speaker_name)
        with open(os.path.join(OUTPUT_DIRECTORY, speaker_name, 'data.txt'), 'r') as f:
            lines = [l for l in f]

        with Pool(64) as p:
            result = p.map(distribution_mapper, lines)

        dist_data[speaker_name] = result

    with open('data.pickle', 'wb') as handle:
        pickle.dump(dist_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Distribution check done!')

    # TODO: Move form_train_val_set here + make cleaning part working with new configs




# Preprocessing function
def process(path, output_directory, speaker_name, speaker_id, process_audio=True):
    cmds = []

    with open(os.path.join(path, 'metadata.csv'), 'r') as file:
        lines = []
        files_to_process = []
        output_path = os.path.join(output_directory, speaker_name)
        output_audio_path = os.path.join(output_path, 'wavs')
        inter_audio_path = os.path.join(output_path, 'wavs_inter')

        pathlib.Path(output_audio_path).mkdir(parents=True, exist_ok=True)
        pathlib.Path(inter_audio_path).mkdir(parents=True, exist_ok=True)

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

            files_to_process.append((input_file_path, inter_file_path, final_file_path, process_audio))

            new_line = '|'.join([final_file_path, text, str(speaker_id)]) + '\n'

            lines.append(new_line)

        with open(os.path.join(output_path, 'data.txt'), 'a+') as f:
            f.writelines(lines)

    return files_to_process

# Cleaning mapper
def cleaning_mapper(job):
    fin, fint, fout, process_audio = job

    if process_audio:
        data, _ = librosa.load(fin, sr=SR)
        data, _ = librosa.effects.trim(data, top_db=TOP_DB)

        wavfile.write(fint, SR, data)

        command = "ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar {} {}".format(fint, SR, fout)
        os.system(command)
    else:
        copyfile(fin, fout)


# Disctribution mapper
def distribution_mapper(line):
    fp, text, _ = line.strip().split('|')

    seq = text_to_sequence(text, ['english_cleaners'])


    if os.path.isfile(fp):
        with audioread.audio_open(fp) as f:
            duration = f.duration
    else:
        duration = None

    return fp, len(seq), duration













if __name__ == '__main__':
    main(OUTPUT_DIRECTORY, DATA_CONFIG)
