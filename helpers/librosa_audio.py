import os
import librosa
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

wavs_in_path = "/workspace/data/aws/dataset/samantha/wavs_in"
wavs_out_path = "/workspace/data/aws/dataset/samantha/wavs_trimed"

sr = 22050
top_db = 40


wavs = os.listdir(wavs_in_path)

for wav in tqdm(wavs):
    data, _ = librosa.load(os.path.join(wavs_in_path, wav), sr=sr)

    data, _ = librosa.effects.trim(data, top_db=top_db)

    wavfile.write(os.path.join(wavs_out_path, wav), sr, data)
