# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import os

import librosa
import numpy as np
import tensorflow as tf

from model import WaveNetModel, mu_law_encode, mu_law_decode
from datasets.data_feeder import DataFeeder
from hparams import hparams


LOGDIR = './logdir'
SAVE_EVERY = None
TEMPERATURE = 1.0


def get_arguments():

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        'checkpoint', type=str, help='Which model checkpoint to generate from')
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--temperature',
        type=_ensure_positive_float,
        default=TEMPERATURE,
        help='Sampling temperature')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')
    parser.add_argument(
        '--eval_txt',
        type=str,
        default="~/Downloads/training/eavl.txt",
        help="the eval txt"
    )

    arguments = parser.parse_args()
    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,
                sample_rate,
                quantization_channels,
                window_size):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)

    quantized = mu_law_encode(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]


def main():
    args = get_arguments()

    sess = tf.Session()

    net = WaveNetModel(
        batch_size=1,
        dilations=hparams.dilations,
        filter_width=hparams.filter_width,
        residual_channels=hparams.residual_channels,
        dilation_channels=hparams.dilation_channels,
        skip_channels=hparams.skip_channels,
        quantization_channels=hparams.quantization_channels,
        use_biases=hparams.use_biases,
        scalar_input=hparams.scalar_input,
        initial_filter_width=hparams.initial_filter_width,
        local_condition_channel=hparams.local_condition_dim
    )

    eval_list = []
    with open(args.eval_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            eval_list.append(line.strip())

    local_condition_data = np.load(eval_list[0])
    local_condition = []
    for i in range(local_condition_data.shape[0]):
        for _ in range(int(hparams.frame_period * hparams.sample_rate / 1000)):
            local_condition.append(local_condition_data[i, 0:hparams.local_condition_dim])
    local_condition = np.asarray(local_condition)

    samples = tf.placeholder(tf.int32)
    local_ph = tf.placeholder(tf.float32, shape=(1, hparams.local_condition_dim))

    next_sample = net.predict_proba_incremental(samples, local_ph)

    sess.run(tf.global_variables_initializer())
    sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    decode = mu_law_decode(samples, hparams.quantization_channels)

    quantization_channels = hparams.quantization_channels

    if args.wav_seed:
        seed = create_seed(args.wav_seed,
                           hparams.sample_rate,
                           quantization_channels,
                           net.receptive_field)
        waveform = sess.run(seed).tolist()
    else:
        # Silence with a single random sample at the end.
        waveform = [quantization_channels / 2] * (net.receptive_field - 1)
        waveform.append(np.random.randint(quantization_channels))

    if args.wav_seed:
        outputs = [next_sample]
        outputs.extend(net.push_ops)

        print('Priming generation...')
        for i, x in enumerate(waveform[-net.receptive_field: -1]):
            if i % 100 == 0:
                print('Priming sample {}'.format(i))
            sess.run(outputs, feed_dict={samples: x, local_ph: local_condition[i:i+1, :]})
        print('Done.')

    sample_len = local_condition.shape[0]
    last_sample_timestamp = datetime.now()
    for step in range(sample_len):

        outputs = [next_sample]
        outputs.extend(net.push_ops)
        window = waveform[-1]

            # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window, local_ph: local_condition[step:step+1, :]})[0]

        # Scale prediction distribution using temperature.
        np.seterr(divide='ignore')
        scaled_prediction = np.log(prediction) / args.temperature
        scaled_prediction = (scaled_prediction -
                             np.logaddexp.reduce(scaled_prediction))
        scaled_prediction = np.exp(scaled_prediction)
        np.seterr(divide='warn')

        sample = np.random.choice(
            np.arange(quantization_channels), p=scaled_prediction)
        waveform.append(sample)

        current_sample_timestamp = datetime.now()
        time_since_print = current_sample_timestamp - last_sample_timestamp
        if time_since_print.total_seconds() > 1.:
            print('Sample {:3<d}/{:3<d}'.format(step + 1, sample_len),
                  end='\r')
            last_sample_timestamp = current_sample_timestamp

        # If we have partial writing, save the result so far.
        if (args.wav_out_path and args.save_every and
                        (step + 1) % args.save_every == 0):
            out = sess.run(decode, feed_dict={samples: waveform})
            write_wav(out, hparams.sample_rate, args.wav_out_path)

            # Introduce a newline to clear the carriage return from the progress.
    print()
    # Save the result as a wav file.
    if args.wav_out_path:
        out = sess.run(decode, feed_dict={samples: waveform})
        write_wav(out, hparams.sample_rate, args.wav_out_path)

    print('Finished generating. The result can be viewed in TensorBoard.')


if __name__ == '__main__':
    main()


