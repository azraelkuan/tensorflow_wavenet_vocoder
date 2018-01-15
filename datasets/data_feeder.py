# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import random
import numpy as np
from scipy.io import wavfile
from hparams import hparams


def get_file_list(metadata_filename):
    files = []
    with open(metadata_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("|")
            local_condition_path = line[0]
            wav_path = line[2]
            files.append((wav_path, local_condition_path))
    return files


def randomize_file(files):
    random_file = random.choice(files)
    yield random_file


def load_npy_data(metadata_filename):
    # print("Loading data...", end="")
    files = get_file_list(metadata_filename)
    # print("Done!")
    # print("File length:{}".format(len(files)))
    random_files = randomize_file(files)
    for each in random_files:
        fs, audio = wavfile.read(each[0])
        audio = audio.reshape(-1, 1)

        origin_local_condtion = np.load(each[1])
        expand_local_condition = []
        # do expand
        for i in range(origin_local_condtion.shape[0]):
            for _ in range(int(hparams.frame_period * fs / 1000)):
                expand_local_condition.append(origin_local_condtion[i, 0:hparams.local_condition_dim])

        local_condition = np.array(expand_local_condition)

        yield audio, local_condition


class DataFeeder(object):
    def __init__(self, metadata_filename, coord, receptive_field, sample_size=None, queue_size=32):
        self.metadata_filename = metadata_filename
        self.coord = coord
        self.receptive_field = receptive_field
        self.sample_size = sample_size
        self.queue_size = queue_size

        self.threads = []

        self._placeholders = [
            tf.placeholder(tf.float32, shape=None),
            tf.placeholder(tf.float32, shape=None)
        ]

        self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                         [tf.float32, tf.float32],
                                         shapes=[(None, 1), (None, hparams.local_condition_dim)],
                                         name='input_queue')

        self.enqueue = self.queue.enqueue(self._placeholders)

    def dequeue(self, batch_size):
        output = self.queue.dequeue_many(batch_size)
        return output

    def thread_main(self, sess):
        stop = False
        while not stop:
            iterator = load_npy_data(self.metadata_filename)
            for audio, local_condition in iterator:
                if self.coord.should_stop():
                    stop = True
                    break

                # force to align the audio and local_condition
                if audio.shape[0] > local_condition.shape[0]:
                    audio = audio[:local_condition.shape[0], :]
                else:
                    local_condition = local_condition[:audio.shape[0], :]

                audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]], mode='constant')
                local_condition = np.pad(local_condition, [[self.receptive_field, 0], [0, 0]], mode='constant')

                if self.sample_size:
                    while len(audio) > self.receptive_field:
                        audio_piece = audio[:(self.receptive_field + self.sample_size), :]
                        audio = audio[self.sample_size:, :]

                        local_condition_piece = local_condition[:(self.receptive_field + self.sample_size), :]
                        local_condition = local_condition[self.sample_size:, :]

                        sess.run(self.enqueue, feed_dict=
                        dict(zip(self._placeholders, (audio_piece, local_condition_piece))))
                else:
                    sess.run(self.enqueue, feed_dict=dict(zip(self._placeholders, (audio, local_condition))))

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
