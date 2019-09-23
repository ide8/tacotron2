import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(42)
N = 10000

if __name__ == '__main__':
    speakers = {
        'blizzard_2013': 2,
        #'elizabeth_klett': 5,
        #'judy_bieber': 3,
        'linda_johnson': 0,
        #'mary_ann': 4,
        'samantha_old': 1, #'samantha_fresh': 1,
        #'elliot_miller': 6
    }

    tdata = '/workspace/training_data'
    p = '/workspace/code/tacotron2_multispeaker_pytorch/processing/data.pickle'

    train_lines = []
    val_lines = []

    with open(p, 'rb') as handle:
        data = pickle.load(handle)

    for key in data.keys():
        data[key] = pd.DataFrame(data[key])
        data[key].rename(columns={0: 'path', 1: 'text', 2: 'dur'}, inplace=True)

    max_text = data['linda_johnson']['text'].max()
    min_text = data['linda_johnson']['text'].min()
    max_dur = data['linda_johnson']['dur'].max()
    min_dur = data['linda_johnson']['dur'].min()

    new_data = {}

    for key in data.keys():
        df = data[key]
        new_data[key] = df[(df['text'] <= max_text) & (df['text'] >= 1) & (df['dur'] <= max_dur) & (df['dur'] >= 1)]


    for speaker in tqdm(speakers.keys()):
        df = new_data[speaker]
        msk = np.random.rand(len(df)) < 0.95

        train = df[msk]
        val = df[~msk]

        if len(train) > N:
            train = train.sample(N, random_state=42)

        train_set = set(train['path'])
        val_set = set(val['path'])

        print(speaker)
        if speaker == 'samantha_old':
            print(len(train) * 2)
            print(len(val) * 2)
        else:
            print(len(train))
            print(len(val))


        with open(os.path.join(tdata, speaker, 'data.txt'), 'r') as f:
            for line in f:
                file_path, _, _ = line.strip().split('|')

                if file_path in train_set:
                    train_lines.append(line)

                    if speaker == 'samantha_old':
                        train_lines.append(line)

                if file_path in val_set:
                    val_lines.append(line)

                    if speaker == 'samantha_old':
                        val_lines.append(line)

    with open(os.path.join(tdata, 'train.txt'), 'w') as f:
        f.writelines(train_lines)

    with open(os.path.join(tdata, 'val.txt'), 'w') as f:
        f.writelines(val_lines)
