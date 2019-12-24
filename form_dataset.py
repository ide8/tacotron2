import pandas as pd
from shutil import copyfile
import os
import os.path
import csv

D = pd.read_excel('dialog-acts.xlsx')
print(D.head())
print(D.columns)
output_folder = 'dataset'
input_folder = "Dialogue_Acts/dialogueacts_sadstrong_2"


lines = []
for dirpath, dirnames, filenames in os.walk(input_folder):
    for filename in [f for f in filenames]:
        old_path = os.path.join(dirpath, filename)
        id = filename.strip().split('.')[0]
        if (len(id) > 0):
            #id = int(id)
            #emotion = D[D['index'] == id]['emotion'].unique()[0]
            #intensity = D[D['index'] == id]['intensity'].unique()[0]
            text = D[D['index'] == int(id)]['text'].unique()[0]
            new_filename = id + "Dialogue_Acts_sad_strong" + '.wav'
            #textline = new_filename + str('|') + text + str('|') + emotion + str('_') + intensity
            textline = new_filename + str('|') + text + str('|') + 'sad_strong'
            lines.append(textline)
            new_path = os.path.join(output_folder, new_filename)
            print(old_path)
            print(new_path)
            print(textline)
            print('___________________')
            copyfile(old_path, new_path)

with open(input_folder+'_metadata.csv', 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')
