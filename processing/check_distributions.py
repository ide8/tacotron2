import os
import sys
import audioread
import pickle

from multiprocessing import Pool
from tqdm import tqdm

sys.path.append('/workspace/code/tacotron2_multispeaker_pytorch')
from tacotron2.text import text_to_sequence

speakers = [
    'blizzard_2013',
    'elizabeth_klett',
    'elliot_miller',
    'judy_bieber',
    'linda_johnson',
    'mary_ann',
    'samantha_fresh',
    'samantha_old'
]

data_path = '/workspace/training_data/'


def mapper(line):
    fp, text, _ = line.strip().split('|')

    seq = text_to_sequence(text, ['english_cleaners'])


    if os.path.isfile(fp):
        with audioread.audio_open(fp) as f:
            duration = f.duration
    else:
        duration = None

    return fp, len(seq), duration


if __name__ == '__main__':
    data = {}
    for sp in tqdm(speakers):
        data[sp] = []

        print(sp)
        with open(os.path.join(data_path, sp, 'data.txt'), 'r') as f:
            lines = [l for l in f]

        with Pool(64) as p:
            result = p.map(mapper, lines)

        data[sp] = result

    with open('data.pickle', 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print('Done!')
