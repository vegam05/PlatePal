import tensorflow as tf
from tensorflow.keras.layers import Conv1D, UpSampling1D, Conv3D, UpSampling3D, Embedding
from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.python.keras.utils import conv_utils
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D

class ConvSN2DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', output_padding=None, data_format=None,
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(ConvSN2DTranspose, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.output_padding = output_padding
        self.data_format = data_format
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint
        self.conv2d_transpose = Conv2DTranspose(filters=filters,
                                                kernel_size=kernel_size,
                                                strides=strides,
                                                padding=padding,
                                                output_padding=output_padding,
                                                data_format=data_format,
                                                activation=activation,
                                                use_bias=use_bias,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer,
                                                kernel_regularizer=kernel_regularizer,
                                                bias_regularizer=bias_regularizer,
                                                kernel_constraint=kernel_constraint,
                                                bias_constraint=bias_constraint)

    def call(self, inputs, training=None):
        return self.conv2d_transpose(inputs)


class ConvSN3DTranspose(Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1, 1), padding='valid', output_padding=None, data_format=None,
                 activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros',
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None,
                 bias_constraint=None, **kwargs):
        super(ConvSN3DTranspose, self).__init__(activity_regularizer=activity_regularizer, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.output_padding = output_padding
        self.data_format = data_format
        self.activation = activation
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.kernel_constraint = kernel_constraint
        self.bias_constraint = bias_constraint

    def build(self, input_shape):
        if len(input_shape) != 5:
            raise ValueError('Inputs should have rank ' +
                             str(5) +
                             '; Received input shape:', str(input_shape))
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
        input_dim = input_shape[channel_axis]
        kernel_shape = self.kernel_size + (input_dim, self.filters)

        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None

        self.u = self.add_weight(shape=tuple([1, self.kernel.shape[-1]]),
                                 initializer=initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)

        # Set input spec.
        self.input_spec = InputSpec(ndim=5, axes={channel_axis: input_dim})
        self.built = True

    def call(self, inputs, training=None):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        if self.data_format == 'channels_first':
            d_axis, h_axis, w_axis = 2, 3, 4
        else:
            d_axis, h_axis, w_axis = 1, 2, 3

        depth, height, width = input_shape[d_axis], input_shape[h_axis], input_shape[w_axis]
        kernel_d, kernel_h, kernel_w = self.kernel_size
        stride_d, stride_h, stride_w = self.strides
        if self.output_padding is None:
            out_pad_d = out_pad_h = out_pad_w = None
        else:
            out_pad_d, out_pad_h, out_pad_w = self.output_padding

        out_depth = conv_utils.deconv_length(depth,
                                             stride_d, kernel_d,
                                             self.padding,
                                             out_pad_d)
        out_height = conv_utils.deconv_length(height,
                                              stride_h, kernel_h,
                                              self.padding,
                                              out_pad_h)
        out_width = conv_utils.deconv_length(width,
                                             stride_w, kernel_w,
                                             self.padding,
                                             out_pad_w)
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_depth, out_height, out_width)
        else:
            output_shape = (batch_size, out_depth, out_height, out_width, self.filters)

        # Spectral Normalization
        def _l2normalize(v, eps=1e-12):
            return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(tf.matmul(_u, tf.transpose(W)))
            _u = _l2normalize(tf.matmul(_v, W))
            return _u, _v

        W_shape = self.kernel.shape
        W_reshaped = tf.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        sigma = tf.matmul(_v, W_reshaped)
        sigma = tf.matmul(sigma, tf.transpose(_u))
        W_bar = W_reshaped / sigma
        if training in {0, False}:
            W_bar = tf.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                W_bar = tf.reshape(W_bar, W_shape)
        self.embeddings = W_bar

        out = tf.gather(self.embeddings, inputs)

        return out
