# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
from datetime import datetime

import tensorflow as tf

from model import WaveNetModel, optimizer_factory
from datasets.data_feeder import DataFeeder
from hparams import hparams, hparams_debug_string

# default parameters
BATCH_SIZE = 1
TRAIN_TXT = "./train.txt"
LOGDIR_ROOT = './logdir'
CHECKPOINT_EVERY = 200
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-4
STARTED_DATE_STRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 8000
L2_REGULARIZATION_STRENGTH = 0
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 5
METADATA = False
PRINT_LOSS_EVERY = 50


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--train_txt', type=str, default=TRAIN_TXT,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,
                        help='Whether to store advanced debugging information '
                             '(execution time, memory consumption) for use with '
                             'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=None,
                        help='Directory in which to store the logging '
                             'information for TensorBoard. '
                             'If the model already exists, it will restore '
                             'the state and will continue training. '
                             'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging '
                             'output and generated model. These are stored '
                             'under the dated subdirectory of --logdir_root. '
                             'Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. '
                             'This creates the new model under the dated directory '
                             'in --logdir_root. '
                             'Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                             'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                             'Default: False')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=optimizer_factory.keys(),
                        help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,
                        default=MOMENTUM, help='Specify the momentum to be '
                                               'used by sgd or rmsprop optimizer. Ignored by the '
                                               'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=bool, default=False,
                        help='Whether to store histogram summaries. Default: False')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP,
                        help='Maximum amount of checkpoints that will be kept alive. Default: '
                             + str(MAX_TO_KEEP) + '.')
    parser.add_argument('--num_gpus', type=int, default=4, help="the number of gpu")
    parser.add_argument('--hparams', type=str, default=None, help="the hparams")
    parser.add_argument('--speaker_id', type=int, default=None, help='the speaker id')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),
          end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path
                          .split('/')[-1]
                          .split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step, sess
    else:
        print(" No checkpoint found.")
        return None, sess


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATE_STRING)
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            if g is None:
                continue
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        if len(grads) == 0:
            average_grads.append((None, v))
            continue

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main():
    args = get_arguments()
    # override the hparams
    if args.hparams is not None:
        hparams.parse(args.hparams)
    hparams.global_cardinality = None if hparams.global_cardinality == 0 else hparams.global_cardinality
    hparams.global_channel = None if hparams.global_channel == 0 else hparams.global_channel
    print(hparams_debug_string())

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    is_overwritten_training = logdir != restore_from

    coord = tf.train.Coordinator()

    with tf.name_scope('create_input'):
        reader = DataFeeder(
            metadata_filename=args.train_txt,
            coord=coord,
            receptive_field=WaveNetModel.calculate_receptive_field(
                hparams.filter_width,
                hparams.dilations
            ),
            gc_enable=hparams.gc_enable,
            sample_size=args.sample_size,
            npy_dataroot=hparams.NPY_DATAROOT,
            num_mels=hparams.num_mels,
            speaker_id=args.speaker_id
        )

    net = WaveNetModel(
        batch_size=args.batch_size,
        dilations=hparams.dilations,
        filter_width=hparams.filter_width,
        residual_channels=hparams.residual_channels,
        dilation_channels=hparams.dilation_channels,
        skip_channels=hparams.skip_channels,
        out_channels=hparams.out_channels,
        use_biases=hparams.use_biases,
        scalar_input=hparams.scalar_input,
        histograms=args.histograms,
        local_condition_channel=hparams.num_mels,
        upsample_conditional_features=hparams.upsample_conditional_features,
        upsample_factor=hparams.upsample_factor,
        global_cardinality=hparams.global_cardinality,
        global_channel=hparams.global_channel
    )

    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None

    trainable = tf.trainable_variables()

    # get global step
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # decay learning rate
    # Calculate the learning rate schedule.
    # decay_steps = hparams.NUM_STEPS_RATIO_PER_DECAY * args.num_steps
    # # Decay the learning rate exponentially based on the number of steps.
    # lr = tf.train.exponential_decay(args.learning_rate,
    #                                 global_step,
    #                                 decay_steps,
    #                                 hparams.LEARNING_RATE_DECAY_FACTOR,
    #                                 staircase=True)

    optimizer = optimizer_factory[args.optimizer](
        learning_rate=args.learning_rate,
        momentum=args.momentum)

    mul_batch_size = args.batch_size * args.num_gpus
    if hparams.gc_enable:
        audio_batch, lc_batch, gc_batch = reader.dequeue(mul_batch_size)
    else:
        audio_batch, lc_batch = reader.dequeue(mul_batch_size)
        gc_batch = None

    split_audio_batch = tf.split(value=audio_batch, num_or_size_splits=args.num_gpus, axis=0)
    split_lc_batch = tf.split(value=lc_batch, num_or_size_splits=args.num_gpus, axis=0)
    if hparams.gc_enable:
        split_gc_batch = tf.split(value=gc_batch, num_or_size_splits=args.num_gpus, axis=0)
    else:
        split_gc_batch = [None for _ in range(args.num_gpus)]

    # support multi gpu train
    tower_grads = []
    tower_losses = []
    with tf.variable_scope(tf.get_variable_scope()):
        for i in range(args.num_gpus):
            with tf.device('/gpu:{}'.format(i)):
                with tf.name_scope('losstower_{}'.format(i)) as scope:
                    loss = net.loss(input_batch=split_audio_batch[i],
                                    local_condition=split_lc_batch[i],
                                    global_condition=split_gc_batch[i],
                                    l2_regularization_strength=args.l2_regularization_strength, name=scope)
                    tf.get_variable_scope().reuse_variables()
                    tower_losses.append(loss)
                    grad_vars = optimizer.compute_gradients(loss, var_list=trainable)
                    tower_grads.append(grad_vars)
    if args.num_gpus == 1:
        optim = optimizer.minimize(loss, var_list=trainable, global_step=global_step)
    else:
        loss = tf.reduce_mean(tower_losses)
        avg_grad = average_gradients(tower_grads)
        optim = optimizer.apply_gradients(avg_grad, global_step=global_step)

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        hparams.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    train_op = tf.group(optim, variables_averages_op)

    # init the sess
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                                                       gpu_options=tf.GPUOptions(allow_growth=True)))
    init = tf.global_variables_initializer()
    sess.run(init)

    saver = tf.train.Saver(var_list=tf.trainable_variables(), max_to_keep=args.max_checkpoints)

    try:
        saved_global_step, sess = load(saver, sess, restore_from)
        if is_overwritten_training or saved_global_step is None:
            saved_global_step = 0
    except:
        print("Something went wrong while restoring checkpoint. "
              "We will terminate training to avoid accidentally overwriting "
              "the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step

    try:
        print_loss = 0.
        start_time = time.time()
        for step in range(saved_global_step, args.num_steps):

            loss_value, _ = sess.run([loss, train_op])

            print_loss += loss_value

            if step % PRINT_LOSS_EVERY == 0:
                duration = time.time() - start_time
                print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(
                    step, print_loss / PRINT_LOSS_EVERY, duration / PRINT_LOSS_EVERY))
                start_time = time.time()
                print_loss = 0.

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
