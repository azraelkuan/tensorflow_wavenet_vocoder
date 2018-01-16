# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import os
import audio

import librosa
import numpy as np
import tensorflow as tf

from model import WaveNetModel
from hparams import hparams
import nnmnkwii.preprocessing as P

from tqdm import tqdm

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
    parser.add_argument(
        '--hparams',
        type=str,
        default=None,
        help="the override hparams"
    )
    parser.add_argument(
        '--gc_id',
        type=int,
        default=0,
        help='the global condition'
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

    # quantized = mu_law_encode(audio, quantization_channels)
    quantized = P.mulaw_quantize(audio, quantization_channels)
    cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size),
                        lambda: tf.size(quantized),
                        lambda: tf.constant(window_size))

    return quantized[:cut_index]


def main():
    args = get_arguments()
    if args.hparams is not None:
        hparams.parse(args.hparams)
    hparams.global_cardinality = None if hparams.global_cardinality == 0 else hparams.global_cardinality
    hparams.global_channel = None if hparams.global_channel == 0 else hparams.global_channel
    print(hparams)

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True)))

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
        local_condition_channel=hparams.num_mels,
        global_cardinality=hparams.global_cardinality,
        global_channel=hparams.global_channel
    )

    local_condition = None
    if hparams.global_channel is not None:
        gc_id = args.gc_id
    else:
        gc_id = None

    with open(args.eval_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[0:1]:
            if line is not None:
                line = line.strip().split('|')
                npy_path = os.path.join(hparams.NPY_DATAROOT, line[1])
                local_condition = np.load(npy_path)

    upsample_factor = audio.get_hop_size()

    # Tc = local_condition.shape[0]
    # length = Tc * upsample_factor
    if hparams.upsample_conditional_features:
        local_condition = np.repeat(local_condition, upsample_factor, axis=0)

    samples = tf.placeholder(tf.int32)
    local_ph = tf.placeholder(tf.float32, shape=(1, hparams.num_mels))

    next_sample = net.predict_proba_incremental(samples, local_ph, gc_id)

    sess.run(tf.global_variables_initializer())
    sess.run(net.init_ops)

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    # decode = mu_law_decode(samples, hparams.quantization_channels)

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

    if args.wav_seed:
        begin_len = net.receptive_field
    else:
        begin_len = 0

    sample_len = local_condition.shape[0]
    for step in tqdm(range(begin_len, sample_len)):

        outputs = [next_sample]
        outputs.extend(net.push_ops)
        window = waveform[-1]

        # Run the WaveNet to predict the next sample.
        prediction = sess.run(outputs, feed_dict={samples: window,
                                                  local_ph: local_condition[step:step+1, :]})[0]

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

        # If we have partial writing, save the result so far.
        if (args.wav_out_path and args.save_every and
                        (step + 1) % args.save_every == 0):
            out = P.inv_mulaw_quantize(np.array(waveform), quantization_channels)
            write_wav(out, hparams.sample_rate, args.wav_out_path)

            # Introduce a newline to clear the carriage return from the progress.
    print()
    # Save the result as a wav file.
    if args.wav_out_path:
        out = P.inv_mulaw_quantize(np.array(waveform), quantization_channels)
        write_wav(out, hparams.sample_rate, args.wav_out_path)

    print('Finished generating.')


if __name__ == '__main__':
    main()


