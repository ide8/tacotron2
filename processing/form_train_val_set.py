import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(42)
minimum_viable_dur = 0.05

N = 100000

if __name__ == '__main__':
    speakers = {
        #'blizzard_2013': 2,
        #'elizabeth_klett': 5,
        #'judy_bieber': 3,
        'linda_johnson': 0,
        #'mary_ann': 4,
        #'samantha_default': 1,
        #'scarjo_her': 1,
        #'scarjo_the_dive_descript_grouped': 1,
        #'scarjo_the_dive_descript_ungrouped': 1,
        #'elliot_miller': 6
    }

    limit_by = 'linda_johnson'
    tdata = '/workspace/training_data'
    p = '/workspace/code/gst/processing/data.pickle'

    train_lines = []
    val_lines = []

    with open(p, 'rb') as handle:
        data = pickle.load(handle)

    for key in data.keys():
        data[key] = pd.DataFrame(data[key])
        data[key].rename(columns={0: 'path', 1: 'text', 2: 'dur'}, inplace=True)

    print('Speakers info')

    for speaker in speakers.keys():
        mint = data[speaker]['text'].min()
        maxt = data[speaker]['text'].max()
        mind = data[speaker]['dur'].min()
        maxd = data[speaker]['dur'].max()

        if speaker == limit_by:
            min_text = mint
            max_text = maxt
            min_dur = mind
            max_dur = maxd

        print('Min {} text: {}'.format(speaker, mint))
        print('Max {} text: {}'.format(speaker, maxt))
        print('Min {} dur: {}'.format(speaker, mind))
        print('Max {} dur: {}'.format(speaker, maxd))
        print('-----------------------------------------')

    new_data = {}

    for key in data.keys():
        df = data[key]
        new_data[key] = df[(df['text'] <= max_text) & (df['text'] >= 1) & (df['dur'] <= max_dur) & (df['dur'] >= minimum_viable_dur)]

    print('Speakers data len')
    for speaker in tqdm(speakers.keys()):
        df = new_data[speaker]
        msk = np.random.rand(len(df)) < 0.95

        train = df[msk]
        val = df[~msk]

        if len(train) > N:
            train = train.sample(N, random_state=42)

        train_set = set(train['path'])
        val_set = set(val['path'])

        print('Train set for {} is: {}'.format(speaker, len(train)))
        print('Val set for {} is: {}'.format(speaker, len(val)))
        print('-----------------------------------------')

        with open(os.path.join(tdata, speaker, 'data.txt'), 'r') as f:
            for line in f:
                file_path, _, _ = line.strip().split('|')

                if file_path in train_set:
                    train_lines.append(line)

                if file_path in val_set:
                    val_lines.append(line)

    with open(os.path.join(tdata, 'train.txt'), 'w') as f:
        f.writelines(train_lines)

    with open(os.path.join(tdata, 'val.txt'), 'w') as f:
        f.writelines(val_lines)
