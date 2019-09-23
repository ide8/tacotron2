import os
import time
import pathlib
import librosa
from tqdm import tqdm
from shutil import copyfile
from scipy.io import wavfile
from multiprocessing import Pool

SR = 22050
TOP_DB = 40


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
        p.map(mapper, jobs)

    print('Done!')


def process(path, output_directory, speaker_name, speaker_id, process_audio=True):
    # print('---------------------------------------------------------------------------------')
    # print('path:', path)
    # print('speaker_name:', speaker_name)
    # print('file_prefix:', file_prefix)
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


def mapper(job):
    fin, fint, fout, process_audio = job

    if process_audio:
        data, _ = librosa.load(fin, sr=SR)
        data, _ = librosa.effects.trim(data, top_db=TOP_DB)

        wavfile.write(fint, SR, data)

        command = "ffmpeg -y -i {} -acodec pcm_s16le -ac 1 -ar {} {}".format(fint, SR, fout)
        os.system(command)
    else:
        copyfile(fin, fout)


if __name__ == '__main__':
    output_directory = '/workspace/training_data'

    data = [
        # (
        #     '/workspace/data/aws/dataset/linda_johnson',
        #     0,
        #     False
        # ),
        (
            '/workspace/data/gcp/samantha_old', #'/workspace/data/aws/dataset/samantha_fresh',
            1,
            True
        ),
        # (
        #     '/workspace/data/aws/dataset/blizzard_2013',
        #     2,
        #     True
        # ),
        # (
        #     '/workspace/data/aws/dataset/en_US/by_book/female/judy_bieber',
        #     3,
        #     True
        # ),
        # (
        #     '/workspace/data/aws/dataset/en_US/by_book/female/mary_ann',
        #     4,
        #     True
        # ),
        # (
        #     '/workspace/data/aws/dataset/en_UK/by_book/female/elizabeth_klett',
        #     5,
        #     True
        # ),
        # (
        #     '/workspace/data/aws/dataset/en_US/by_book/male/elliot_miller',
        #     6,
        #     True
        # )
    ]

    main(output_directory, data)