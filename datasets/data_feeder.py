# -*- coding: utf-8 -*-
import tensorflow as tf
import threading
import random
import numpy as np
import os
import audio


def get_file_list(metadata_filename, npy_dataroot):
    files = []
    with open(metadata_filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip().split("|")
            local_condition_path = line[1]
            wav_path = line[0]
            local_condition_path = os.path.join(npy_dataroot, local_condition_path)
            wav_path = os.path.join(npy_dataroot, wav_path)
            if len(line) == 5:
                global_condition = int(line[4])
            else:
                global_condition = None

            files.append((wav_path, local_condition_path, global_condition))
    return files


def randomize_file(files):
    random_file = random.choice(files)
    yield random_file


def load_npy_data(metadata_filename, npy_dataroot):
    # print("Loading data...", end="")
    files = get_file_list(metadata_filename, npy_dataroot)
    # print("Done!")
    # print("File length:{}".format(len(files)))
    random_files = randomize_file(files)
    for each in random_files:

        wav = np.load(each[0])
        local_condition = np.load(each[1])
        global_condition = each[2]

        wav, local_condition = audio.adjust_time_resolution(wav, local_condition)

        wav = wav.reshape(-1, 1)
        yield wav, local_condition, global_condition


class DataFeeder(object):
    def __init__(self, metadata_filename, coord, receptive_field, gc_enable=False,
                 sample_size=None, queue_size=32, npy_dataroot=None, num_mels=None):
        self.metadata_filename = metadata_filename
        self.coord = coord
        self.receptive_field = receptive_field
        self.sample_size = sample_size
        self.queue_size = queue_size
        self.gc_enable = gc_enable
        self.npy_dataroot = npy_dataroot
        self.num_mels = num_mels

        self.threads = []

        self._placeholders = [
            tf.placeholder(tf.float32, shape=None),
            tf.placeholder(tf.float32, shape=None)
        ]

        if self.gc_enable:
            self._placeholders.append(tf.placeholder(tf.int32, shape=None))
            self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                             [tf.float32, tf.float32, tf.int32],
                                             shapes=[(None, 1), (None, self.num_mels), ()],
                                             name='input_queue')
        else:
            self.queue = tf.PaddingFIFOQueue(self.queue_size,
                                             [tf.float32, tf.float32],
                                             shapes=[(None, 1), (None, self.num_mels)],
                                             name='input_queue')

        self.enqueue = self.queue.enqueue(self._placeholders)

    def dequeue(self, batch_size):
        output = self.queue.dequeue_many(batch_size)
        return output

    def thread_main(self, sess):
        stop = False
        while not stop:
            iterator = load_npy_data(self.metadata_filename, self.npy_dataroot)
            for audio, local_condition, global_condition in iterator:
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

                        if self.gc_enable:
                            sess.run(self.enqueue, feed_dict=
                            dict(zip(self._placeholders, (audio_piece, local_condition_piece, global_condition))))
                        else:
                            sess.run(self.enqueue, feed_dict=
                            dict(zip(self._placeholders, (audio_piece, local_condition_piece))))
                else:
                    if self.gc_enable:
                        sess.run(self.enqueue, feed_dict=dict(zip(
                            self._placeholders, (audio, local_condition, global_condition))))
                    else:
                        sess.run(self.enqueue, feed_dict=dict(zip(self._placeholders, (audio, local_condition))))

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
