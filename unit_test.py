# -*- coding: utf-8 -*-

from datasets.feature import get_mgc_lf0_uv_bap
from datasets.data_feeder import load_npy_data

# features = get_mgc_lf0_uv_bap("/home/kc430/Downloads/LJSpeech-1.0/wavs/LJ050-0268.wav")
# print(features.shape)

for audio, local_condition in load_npy_data("/home/kc430/Downloads/training/train.txt"):
    print(audio.shape)
    print(local_condition.shape)