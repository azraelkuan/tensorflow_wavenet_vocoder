# -*- coding: utf-8 -*-

import argparse
from datetime import datetime
import os
import audio

import librosa
import numpy as np
import tensorflow as tf

from model import WaveNetModel
from hparams import hparams, hparams_debug_string
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
        '--save_every',
        type=int,
        default=SAVE_EVERY,
        help='How many samples before saving in-progress wav')
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
        '--fast',
        type=int,
        default=1
    )
    arguments = parser.parse_args()
    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    # maxv = np.iinfo(np.int16).max
    # librosa.output.write_wav(filename, (y * maxv).astype(np.int16), sample_rate)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def main():
    args = get_arguments()
    if args.hparams is not None:
        hparams.parse(args.hparams)
    if not hparams.gc_enable:
        hparams.global_channel = None
        hparams.global_cardinality = None

    print(hparams_debug_string())

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, gpu_options=tf.GPUOptions(allow_growth=True)))

    net = WaveNetModel(
        batch_size=1,
        dilations=hparams.dilations,
        filter_width=hparams.filter_width,
        residual_channels=hparams.residual_channels,
        dilation_channels=hparams.dilation_channels,
        skip_channels=hparams.skip_channels,
        out_channels=hparams.out_channels,
        use_biases=hparams.use_biases,
        scalar_input=hparams.scalar_input,
        local_condition_channel=hparams.num_mels,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_factor=hparams.upsample_factor,
        global_cardinality=hparams.global_cardinality,
        global_channel=hparams.global_channel
    )
    samples = tf.placeholder(tf.int32)
    local_ph = tf.placeholder(tf.float32, shape=(1, hparams.num_mels))

    sess.run(tf.global_variables_initializer())

    variables_to_restore = {
        var.name[:-2]: var for var in tf.global_variables()
        if not ('state_buffer' in var.name or 'pointer' in var.name)}
    saver = tf.train.Saver(variables_to_restore)

    print('Restoring model from {}'.format(args.checkpoint))
    saver.restore(sess, args.checkpoint)

    tmp_global_condition = None
    upsample_factor = audio.get_hop_size()

    generate_list = []
    with open(args.eval_txt, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line is not None:
                line = line.strip().split('|')
                npy_path = os.path.join(hparams.NPY_DATAROOT, line[1])
                tmp_local_condition = np.load(npy_path).astype(np.float32)
                if len(line) == 5:
                    tmp_global_condition = int(line[4])
                if hparams.global_channel is None:
                    tmp_global_condition = None
                generate_list.append((tmp_local_condition, tmp_global_condition, line[1]))

    for local_condition, global_condition, npy_path in generate_list:
        wav_id = npy_path.split('-mel')[0]
        wav_out_path = "wav/{}_gen.wav".format(wav_id)

        if not hparams.upsample_conditional_features:
            local_condition = np.repeat(local_condition, upsample_factor, axis=0)
        else:
            local_condition = np.expand_dims(local_condition, 0)
            local_condition = net.create_upsample(local_condition)
            local_condition = tf.squeeze(local_condition, [0]).eval(session=sess)

        if args.fast:
            next_sample = net.predict_proba_incremental(samples, local_ph, global_condition)
            sess.run(net.init_ops)
        else:
            next_sample = net.predict_proba(samples, local_ph, global_condition)

        quantization_channels = hparams.quantize_channels

        # Silence with a single random sample at the end.
        if hparams.scalar_input:
            waveform = [0] * (net.receptive_field - 1)
            waveform.append(0)
        else:
            waveform = [quantization_channels / 2] * (net.receptive_field - 1)
            waveform.append(np.random.randint(quantization_channels))

        sample_len = local_condition.shape[0]
        for step in tqdm(range(0, sample_len)):
            if args.fast:
                outputs = [next_sample]
                outputs.extend(net.push_ops)
                window = waveform[-1]
            else:
                if len(waveform) > net.receptive_field:
                    window = waveform[-net.receptive_field:]
                else:
                    window = waveform
                outputs = [next_sample]

            # print(window)
            # Run the WaveNet to predict the next sample.
            prediction = sess.run(outputs, feed_dict={samples: window,
                                                      local_ph: local_condition[step:step+1, :]
                                                      })[0]

            # Scale prediction distribution using temperature.
            if hparams.scalar_input:
                # print(prediction)
                # input()
                waveform.append(prediction[-1])
                print(prediction[-1])
                # if abs(prediction[-1]) > 0.1:
                #     print(prediction[-1])
            else:
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
            # if (wav_out_path and args.save_every and
            #                 (step + 1) % args.save_every == 0):
            #     out = P.inv_mulaw_quantize(np.array(waveform), quantization_channels)
            #     write_wav(out, hparams.sample_rate, wav_out_path)

                # Introduce a newline to clear the carriage return from the progress.
        print()
        # Save the result as a wav file.
        if wav_out_path:
            if hparams.scalar_input:
                out = waveform
            else:
                out = P.inv_mulaw_quantize(np.array(waveform).astype(np.int16), quantization_channels)
            write_wav(out, hparams.sample_rate, wav_out_path)
    print('Finished generating.')


if __name__ == '__main__':
    main()


