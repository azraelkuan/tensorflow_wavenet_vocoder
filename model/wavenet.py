# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from .ops import causal_conv


def create_variable(name, shape):
    initializer = tf.contrib.layers.xavier_initializer_conv2d()
    variable = tf.get_variable(name=name, shape=shape, initializer=initializer)
    return variable


def create_bias_variable(name, shape):
    initializer = tf.constant_initializer(value=0.0, dtype=tf.float32)
    return tf.get_variable(name=name, shape=shape, initializer=initializer)


def create_embedding_table(name, shape):
    if shape[0] == shape[1]:
        initial_val = np.identity(n=shape[0], dtype=np.float32)
        return tf.get_variable(name=name, initializer=initial_val)
    else:
        return create_variable(name, shape)


class WaveNetModel(object):

    def __init__(self,
                 batch_size,
                 dilations,
                 filter_width,
                 residual_channels,
                 dilation_channels,
                 skip_channels,
                 quantization_channels=2 ** 8,
                 use_biases=False,
                 scalar_input=False,
                 initial_filter_width=32,
                 histograms=False,
                 local_condition_channel=None,
                 upsample_conditional_features=True,
                 upsample_factor=None,
                 global_cardinality=None,
                 global_channel=None):
        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width
        self.histograms = histograms
        self.local_condition_channel = local_condition_channel
        self.global_channel = global_channel
        self.global_cardinality = global_cardinality
        self.upsample_conditional_features = upsample_conditional_features
        self.upsample_factor = upsample_factor

        self.receptive_field = WaveNetModel.calculate_receptive_field(
            self.filter_width, self.dilations, self.scalar_input,
            self.initial_filter_width)
        self.variables = self._create_variables()

    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input,
                                  initial_filter_width):
        receptive_field = (filter_width - 1) * sum(dilations) + 1
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1
        return receptive_field

    def _create_variables(self):
        var = dict()

        with tf.variable_scope('wavenet'):

            if self.upsample_conditional_features:
                with tf.variable_scope('upsample_layer'):
                    layer = dict()
                    for i in range(len(self.upsample_factor)):
                        layer['upsample{}'.format(i)] = \
                            create_variable('upsample{}'.format(i), [self.upsample_factor[i], self.filter_width, 1, 1])
                    var['upsample_layer'] = layer

            if self.global_cardinality is not None:
                with tf.variable_scope('embeddings'):
                    layer = dict()
                    layer['gc_embedding'] = create_embedding_table(
                        'gc_embedding',
                        [self.global_cardinality, self.global_channel]
                    )
                    var['embeddings'] = layer

            with tf.variable_scope('causal_layer'):
                layer = dict()
                if self.scalar_input:
                    initial_channels = 1
                    initial_filter_width = self.initial_filter_width
                else:
                    initial_channels = self.quantization_channels
                    initial_filter_width = self.filter_width
                layer['filter'] = create_variable(
                    'filter',
                    [initial_filter_width,
                     initial_channels,
                     self.residual_channels])
                var['causal_layer'] = layer

            var['dilated_stack'] = list()
            with tf.variable_scope('dilated_stack'):
                for i, dilation in enumerate(self.dilations):
                    with tf.variable_scope('layer{}'.format(i)):
                        current = dict()
                        current['filter'] = create_variable(
                            'filter',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['gate'] = create_variable(
                            'gate',
                            [self.filter_width,
                             self.residual_channels,
                             self.dilation_channels])
                        current['dense'] = create_variable(
                            'dense',
                            [1,
                             self.dilation_channels,
                             self.residual_channels])
                        current['skip'] = create_variable(
                            'skip',
                            [1,
                             self.dilation_channels,
                             self.skip_channels])

                        if self.local_condition_channel is not None:
                            current['lc_gate_weights'] = create_variable(
                                'lc_gate',
                                [1, self.local_condition_channel,
                                 self.dilation_channels])
                            current['lc_filter_weights'] = create_variable(
                                'lc_filter',
                                [1, self.local_condition_channel,
                                 self.dilation_channels])

                        if self.global_channel is not None:
                            current['gc_gate_weights'] = create_variable(
                                'gc_gate',
                                [1, self.global_channel, self.dilation_channels]
                            )
                            current['gc_filter_weights'] = create_variable(
                                'gc_filter',
                                [1, self.global_channel, self.dilation_channels]
                            )

                        if self.use_biases:
                            current['filter_bias'] = create_bias_variable(
                                'filter_bias',
                                [self.dilation_channels])
                            current['gate_bias'] = create_bias_variable(
                                'gate_bias',
                                [self.dilation_channels])
                            current['dense_bias'] = create_bias_variable(
                                'dense_bias',
                                [self.residual_channels])
                            current['skip_bias'] = create_bias_variable(
                                'slip_bias',
                                [self.skip_channels])

                        var['dilated_stack'].append(current)
                with tf.variable_scope('postprocessing'):
                    current = dict()
                    current['postprocess1'] = create_variable(
                        'postprocess1',
                        [1, self.skip_channels, self.skip_channels])
                    current['postprocess2'] = create_variable(
                        'postprocess2',
                        [1, self.skip_channels, self.quantization_channels])
                    if self.use_biases:
                        current['postprocess1_bias'] = create_bias_variable(
                            'postprocess1_bias',
                            [self.skip_channels])
                        current['postprocess2_bias'] = create_bias_variable(
                            'postprocess2_bias',
                            [self.quantization_channels])
                    var['postprocessing'] = current

            return var

    def _create_causal_layer(self, input_batch):

        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            return causal_conv(input_batch, weights_filter, 1)

    def _create_dilation_layer(self, input_batch, layer_index, dilation,
                               local_condition_batch, global_condition_batch, output_width):

        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']

        conv_filter = causal_conv(input_batch, weights_filter, dilation)
        conv_gate = causal_conv(input_batch, weights_gate, dilation)

        if local_condition_batch is not None:
            weights_lc_filter = variables['lc_filter_weights']
            local_filter = tf.nn.conv1d(local_condition_batch,
                                        weights_lc_filter,
                                        stride=1,
                                        padding="SAME",
                                        name="lc_filter")
            weights_lc_gate = variables['lc_gate_weights']
            local_gate = tf.nn.conv1d(local_condition_batch,
                                      weights_lc_gate,
                                      stride=1,
                                      padding="SAME",
                                      name="lc_gate")

            is_cut = tf.greater(tf.shape(local_filter)[1], tf.shape(conv_filter)[1])
            local_filter = tf.cond(is_cut,
                                   lambda: tf.slice(local_filter, [0, 0, 0], [-1, tf.shape(conv_filter)[1], -1]),
                                   lambda: local_filter)
            local_gate = tf.cond(is_cut, lambda: tf.slice(local_gate, [0, 0, 0], [-1, tf.shape(conv_gate)[1], -1]),
                                 lambda: local_gate)

            conv_filter += local_filter
            conv_gate += local_gate

        if global_condition_batch is not None:
            weights_gc_filter = variables['gc_filter_weights']
            conv_filter += tf.nn.conv1d(global_condition_batch,
                                        weights_gc_filter,
                                        stride=1,
                                        padding="SAME",
                                        name="gc_filter")
            weights_gc_gate = variables['gc_gate_weights']
            conv_gate += tf.nn.conv1d(global_condition_batch,
                                      weights_gc_gate,
                                      stride=1,
                                      padding='SAME',
                                      name='gc_gate')

        if self.use_biases:
            filter_bias = variables['filter_bias']
            gate_bias = variables['gate_bias']
            conv_filter = tf.add(conv_filter, filter_bias)
            conv_gate = tf.add(conv_gate, gate_bias)

        out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)

        # The 1x1 conv to produce the residual output
        weights_dense = variables['dense']
        transformed = tf.nn.conv1d(
            out, weights_dense, stride=1, padding="SAME", name="dense")

        # The 1x1 conv to produce the skip output
        skip_cut = tf.shape(out)[1] - output_width
        out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, -1])
        weights_skip = variables['skip']
        skip_contribution = tf.nn.conv1d(
            out_skip, weights_skip, stride=1, padding="SAME", name="skip")

        if self.use_biases:
            dense_bias = variables['dense_bias']
            skip_bias = variables['skip_bias']
            transformed = transformed + dense_bias
            skip_contribution = skip_contribution + skip_bias

        if self.histograms:
            layer = 'layer{}'.format(layer_index)
            tf.summary.histogram(layer + '_filter', weights_filter)
            tf.summary.histogram(layer + '_gate', weights_gate)
            tf.summary.histogram(layer + '_dense', weights_dense)
            tf.summary.histogram(layer + '_skip', weights_skip)
            if self.use_biases:
                tf.summary.histogram(layer + '_biases_filter', filter_bias)
                tf.summary.histogram(layer + '_biases_gate', gate_bias)
                tf.summary.histogram(layer + '_biases_dense', dense_bias)
                tf.summary.histogram(layer + '_biases_skip', skip_bias)

        input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
        input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])

        return skip_contribution, input_batch + transformed

    def _generator_conv(self, input_batch, state_batch, weights):
        """Perform convolution for a single convolutional processing step."""
        # TODO generalize to filter_width > 2
        past_weights = weights[0, :, :]
        curr_weights = weights[1, :, :]
        output = tf.matmul(state_batch, past_weights) + tf.matmul(
            input_batch, curr_weights)
        return output

    def _generator_causal_layer(self, input_batch, state_batch):
        with tf.name_scope('causal_layer'):
            weights_filter = self.variables['causal_layer']['filter']
            output = self._generator_conv(
                input_batch, state_batch, weights_filter)
        return output

    def _generator_dilation_layer(self, input_batch, state_batch, layer_index,
                                  dilation, local_condition, global_condition):
        variables = self.variables['dilated_stack'][layer_index]

        weights_filter = variables['filter']
        weights_gate = variables['gate']
        output_filter = self._generator_conv(
            input_batch, state_batch, weights_filter)
        output_gate = self._generator_conv(
            input_batch, state_batch, weights_gate)

        if local_condition is not None:
            weights_lc_filter = variables['lc_filter_weights']
            weights_lc_gate = variables['lc_gate_weights']
            local_filter = tf.nn.conv1d(local_condition,
                                        weights_lc_filter,
                                        padding='SAME',
                                        stride=1)
            local_gate = tf.nn.conv1d(local_condition,
                                      weights_lc_gate,
                                      padding='SAME',
                                      stride=1)
            local_filter = tf.squeeze(local_filter, [1])
            local_gate = tf.squeeze(local_gate, [1])
            output_filter += local_filter
            output_gate += local_gate
        if global_condition is not None:
            global_condition = tf.reshape(global_condition, shape=(1, 1, -1))

            weights_gc_filter = variables['gc_filter_weights']
            # weights_gc_filter = weights_gc_filter[0, :, :]
            # output_filter += tf.matmul(global_condition, weights_gc_filter)
            #
            weights_gc_gate = variables['gc_gate_weights']
            # weights_gc_gate = weights_gc_gate[0, :, :]
            # output_gate += tf.matmul(global_condition, weights_gc_gate)
            global_filter = tf.nn.conv1d(global_condition,
                                         weights_gc_filter,
                                         padding='SAME',
                                         stride=1)
            global_gate = tf.nn.conv1d(global_condition,
                                       weights_gc_gate,
                                       padding='SAME',
                                       stride=1)
            global_filter = tf.squeeze(global_filter, [1])
            global_gate = tf.squeeze(global_gate, [1])
            output_filter += global_filter
            output_gate += global_gate

        if self.use_biases:
            output_filter = output_filter + variables['filter_bias']
            output_gate = output_gate + variables['gate_bias']

        out = tf.tanh(output_filter) * tf.sigmoid(output_gate)

        weights_dense = variables['dense']
        transformed = tf.matmul(out, weights_dense[0, :, :])
        if self.use_biases:
            transformed = transformed + variables['dense_bias']

        weights_skip = variables['skip']
        skip_contribution = tf.matmul(out, weights_skip[0, :, :])
        if self.use_biases:
            skip_contribution = skip_contribution + variables['skip_bias']

        return skip_contribution, input_batch + transformed

    def create_upsample(self, local_condition_batch):
        layer_filter = self.variables['upsample_layer']
        local_condition_batch = tf.expand_dims(local_condition_batch, [3])
        # local condition batch N H W C
        batch_size = tf.shape(local_condition_batch)[0]
        upsample_dim = tf.shape(local_condition_batch)[1]

        for i in range(len(self.upsample_factor)):
            upsample_dim = upsample_dim * self.upsample_factor[i]
            output_shape = tf.stack([batch_size, upsample_dim, tf.shape(local_condition_batch)[2], 1])
            local_condition_batch = tf.nn.conv2d_transpose(
                local_condition_batch,
                layer_filter['upsample{}'.format(i)],
                strides=[1, self.upsample_factor[i], 1, 1],
                output_shape=output_shape
            )

        local_condition_batch = tf.squeeze(local_condition_batch, [3])
        return local_condition_batch

    def _create_network(self, input_batch, local_condition_batch, global_condition_batch):
        """Construct the WaveNet network."""
        outputs = []
        current_layer = input_batch

        # Pre-process the input with a regular convolution
        if self.scalar_input:
            initial_channels = 1
        else:
            initial_channels = self.quantization_channels

        current_layer = self._create_causal_layer(current_layer)

        output_width = tf.shape(input_batch)[1] - self.receptive_field + 1

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    output, current_layer = self._create_dilation_layer(
                        current_layer, layer_index, dilation,
                        local_condition_batch, global_condition_batch, output_width)
                    outputs.append(output)

        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = self.variables['postprocessing']['postprocess1']
            w2 = self.variables['postprocessing']['postprocess2']
            if self.use_biases:
                b1 = self.variables['postprocessing']['postprocess1_bias']
                b2 = self.variables['postprocessing']['postprocess2_bias']

            if self.histograms:
                tf.summary.histogram('postprocess1_weights', w1)
                tf.summary.histogram('postprocess2_weights', w2)
                if self.use_biases:
                    tf.summary.histogram('postprocess1_biases', b1)
                    tf.summary.histogram('postprocess2_biases', b2)

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)
            conv1 = tf.nn.conv1d(transformed1, w1, stride=1, padding="SAME")
            if self.use_biases:
                conv1 = tf.add(conv1, b1)
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.nn.conv1d(transformed2, w2, stride=1, padding="SAME")
            if self.use_biases:
                conv2 = tf.add(conv2, b2)

        return conv2

    def _create_generator(self, input_batch, local_condition_batch, global_condition_batch):
        """Construct an efficient incremental generator."""
        init_ops = []
        push_ops = []
        outputs = []
        current_layer = input_batch

        q = tf.FIFOQueue(
            1,
            dtypes=tf.float32,
            shapes=(self.batch_size, self.quantization_channels))
        init = q.enqueue_many(
            tf.zeros((1, self.batch_size, self.quantization_channels)))

        current_state = q.dequeue()
        push = q.enqueue([current_layer])
        init_ops.append(init)
        push_ops.append(push)

        current_layer = self._generator_causal_layer(
            current_layer, current_state)

        # Add all defined dilation layers.
        with tf.name_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations):
                with tf.name_scope('layer{}'.format(layer_index)):
                    q = tf.FIFOQueue(
                        dilation,
                        dtypes=tf.float32,
                        shapes=(self.batch_size, self.residual_channels))
                    init = q.enqueue_many(
                        tf.zeros((dilation, self.batch_size,
                                  self.residual_channels)))

                    current_state = q.dequeue()
                    push = q.enqueue([current_layer])
                    init_ops.append(init)
                    push_ops.append(push)

                    output, current_layer = self._generator_dilation_layer(
                        current_layer, current_state, layer_index, dilation,
                        local_condition_batch, global_condition_batch)
                    outputs.append(output)
        self.init_ops = init_ops
        self.push_ops = push_ops

        with tf.name_scope('postprocessing'):
            variables = self.variables['postprocessing']
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
            w1 = variables['postprocess1']
            w2 = variables['postprocess2']
            if self.use_biases:
                b1 = variables['postprocess1_bias']
                b2 = variables['postprocess2_bias']

            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)
            transformed1 = tf.nn.relu(total)

            conv1 = tf.matmul(transformed1, w1[0, :, :])
            if self.use_biases:
                conv1 = conv1 + b1
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.matmul(transformed2, w2[0, :, :])
            if self.use_biases:
                conv2 = conv2 + b2

        return conv2

    def _one_hot(self, input_batch):
        """
        One-hot encodes the waveform amplitudes.
        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        """
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(
                input_batch,
                depth=self.quantization_channels,
                dtype=tf.float32)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)
        return encoded

    def _embed_gc(self, global_condition):
        embedding = None

        if self.global_cardinality is not None:
            embedding_table = self.variables['embeddings']['gc_embedding']
            embedding = tf.nn.embedding_lookup(embedding_table, global_condition)
        elif global_condition is not None:
            gc_batch_rank = len(global_condition.get_shape())
            dims_batch = (global_condition.get_shape()[gc_batch_rank - 1] == self.global_channel)
            if not dims_batch:
                raise ValueError("shape of global condition not match global_channels")

            embedding = global_condition
        if embedding is not None:
            embedding = tf.reshape(
                embedding, [self.batch_size, 1, self.global_channel]
            )
        return embedding

    def predict_proba_incremental(self, waveform, local_condition=None,
                                  global_condition=None,
                                  name='wavenet'):
        """
        Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.
        """
        if self.filter_width > 2:
            raise NotImplementedError("Incremental generation does not "
                                      "support filter_width > 2.")
        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not "
                                      "support scalar input yet.")
        with tf.name_scope(name):
            encoded = tf.one_hot(waveform, self.quantization_channels)
            encoded = tf.reshape(encoded, [-1, self.quantization_channels])
            local_condition = tf.reshape(local_condition, [1, -1, self.local_condition_channel])
            gc_embedding = self._embed_gc(global_condition)

            raw_output = self._create_generator(encoded, local_condition, gc_embedding)
            out = tf.reshape(raw_output, [-1, self.quantization_channels])
            proba = tf.cast(
                tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)
            last = tf.slice(
                proba,
                [tf.shape(proba)[0] - 1, 0],
                [1, self.quantization_channels])
            return tf.reshape(last, [-1])

    def loss(self,
             input_batch,
             local_condition=None,
             global_condition=None,
             l2_regularization_strength=None,
             name='wavenet'):
        """
        Creates a WaveNet network and returns the autoencoding loss.
        The variables are all scoped to the given name.
        """
        with tf.name_scope(name):
            # We mu-law encode and quantize the input audioform.

            # the input data has already dealed with mu_law
            # encoded_input = mu_law_encode(input_batch,
            #                               self.quantization_channels)

            encoded_input = tf.cast(input_batch, tf.int32)
            gc_embedding = self._embed_gc(global_condition)

            encoded = self._one_hot(encoded_input)
            if self.scalar_input:
                network_input = tf.reshape(
                    tf.cast(input_batch, tf.float32),
                    [self.batch_size, -1, 1])
            else:
                network_input = encoded

            if self.upsample_conditional_features:
                local_condition = self.create_upsample(local_condition)

            assert_op = tf.assert_equal(tf.shape(local_condition)[1], tf.shape(encoded)[1],
                                        data=[local_condition, encoded], name='assert_equal')

            with tf.control_dependencies([assert_op]):

                # Cut off the last sample of network input to preserve causality.
                network_input_width = tf.shape(network_input)[1] - 1
                network_input = tf.slice(network_input, [0, 0, 0],
                                         [-1, network_input_width, -1])

                raw_output = self._create_network(network_input, local_condition, gc_embedding)

                with tf.name_scope('loss'):
                    # Cut off the samples corresponding to the receptive field
                    # for the first predicted sample.
                    target_output = tf.slice(
                        tf.reshape(
                            encoded,
                            [self.batch_size, -1, self.quantization_channels]),
                        [0, self.receptive_field, 0],
                        [-1, -1, -1])
                    target_output = tf.reshape(target_output,
                                               [-1, self.quantization_channels])
                    prediction = tf.reshape(raw_output,
                                            [-1, self.quantization_channels])
                    loss = tf.nn.softmax_cross_entropy_with_logits(
                        logits=prediction,
                        labels=target_output)
                    reduced_loss = tf.reduce_mean(loss)

                    tf.summary.scalar('loss', reduced_loss)

                    if l2_regularization_strength is None:
                        return reduced_loss
                    else:
                        # L2 regularization for all trainable parameters
                        l2_loss = tf.add_n([tf.nn.l2_loss(v)
                                            for v in tf.trainable_variables()
                                            if not ('bias' in v.name)])

                        # Add the regularization term to the loss
                        total_loss = (reduced_loss +
                                      l2_regularization_strength * l2_loss)

                        tf.summary.scalar('l2_loss', l2_loss)
                        tf.summary.scalar('total_loss', total_loss)

                        return total_loss
