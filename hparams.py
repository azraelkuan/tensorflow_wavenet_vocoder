# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    mgc_order=79,
    frame_period=5,
    local_condition_dim=81,
    windows=[
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ],

    filter_width=2,
    sample_rate=16000,
    dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    residual_channels=32,
    dilation_channels=32,
    quantization_channels=256,
    skip_channels=512,
    use_biases=True,
    scalar_input=False,
    initial_filter_width=32

)
