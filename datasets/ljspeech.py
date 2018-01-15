# -*- coding: utf-8 -*-

import numpy as np
import os
from datasets.feature import get_mgc_lf0_uv_bap
from concurrent.futures import ProcessPoolExecutor
from functools import partial


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x:x):
    """
    Pre_processes the LJ Speech dataset from a given input path into a given output directory.
        Args:
          in_dir: The directory where you have downloaded the LJ Speech dataset
          out_dir: The directory to write the output into
          num_workers: Optional number of worker processes to parallelize across
          tqdm: You can optionally pass tqdm to get a nice progress bar
        Returns:
          A list of tuples describing the training examples. This should be written to train.txt
    """
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1
    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            text = parts[2]
            futures.append(executor.submit(partial(_process_utterance, out_dir, index, wav_path, text)))
            index += 1
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
    features = get_mgc_lf0_uv_bap(wav_path)
    n_frames = features.shape[1]
    wav_id = wav_path.split("/")[-1].split(".")[0]
    cmp_file_name = '{}.npy'.format(wav_id)
    np.save(os.path.join(out_dir, cmp_file_name), features, allow_pickle=False)
    return os.path.join(out_dir, cmp_file_name), n_frames, wav_path

