import os
import numpy as np

train = ['Tacotron2/filelists/lj_train.txt', 'filenames_sm/train.txt']
val = ['Tacotron2/filelists/lj_val.txt', 'filenames_sm/val.txt']

path_to = 'lj_sm'

speaker_id = 0

lt = []
for file in train:
    with open(file, 'r') as f:
        train_lines = [l for l in f]

    lt += train_lines

lv = []
for file in val:
    with open(file, 'r') as f:
        val_lines = [l for l in f]

    lv += val_lines



with open(os.path.join(path_to, 'train.txt'), 'w') as f:
    f.writelines(lt)

with open(os.path.join(path_to, 'val.txt'), 'w') as f:
    f.writelines(lv)
