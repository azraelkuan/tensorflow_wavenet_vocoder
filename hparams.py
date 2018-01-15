# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    mgc_order=24,
    frame_period=5,
    local_condition_dim=26,
    windows=[
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ],

    filter_width=2,
    sample_rate=22050,
    dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
               1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    residual_channels=32,
    dilation_channels=32,
    quantization_channels=256,
    skip_channels=512,
    use_biases=True,
    scalar_input=False,
    initial_filter_width=32,

    LEARNING_RATE_DECAY_FACTOR=0.1,
    NUM_STEPS_RATIO_PER_DECAY=0.3,
)
