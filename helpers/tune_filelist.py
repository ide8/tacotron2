import os

fl_path = '/workspace/data/aws/dataset/samantha/filenames'
dataset_path = '/workspace/data/aws/dataset/samantha/waws/'

files = os.listdir(fl_path)

for fn in files:
    if fn.starswith('sm_mel'):
        with open(os.path.join(fl_path, fn), 'r') as f:
            lines = [dataset_path + l for l in f]

        with open(os.path.join(fl_path, fn), 'w') as f:
            f.writelines(lines)
