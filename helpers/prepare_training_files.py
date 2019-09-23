import os
import numpy as np

metadata_path = 'metadata_sm.csv'
folder = 'filenames_sm'

data_path = '/workspace/data/aws/dataset/samantha/wavs/'

low_filter = 2
up_filter = 30

filtered = 0
passed = 0

lines = []

with open(metadata_path, 'r') as f:
    for l in f:
        splt = l.strip().split('|')
        sentence = splt[1]

        c = sentence.count(' ')

        if c >= low_filter and c <= up_filter:
            passed += 1

            lines.append('|'.join(['{}{}.wav'.format(data_path, splt[0]), sentence, '1']) + '\n')
        else:
            filtered += 1

print('Passed:', passed)
print('Filtered:', filtered)


if lines:
    try:
        os.stat(folder)
    except:
        os.mkdir(folder)

    with open(os.path.join(folder, 'train.txt'), 'w') as f:
         f.writelines(lines[:3100])

    with open(os.path.join(folder, 'val.txt'), 'w') as f:
        f.writelines(lines[3100:])

    # with open(os.path.join(folder, 'test.txt'), 'w') as f:
    #     f.writelines(test)




