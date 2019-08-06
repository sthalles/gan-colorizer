# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Keras convolution layers and image transformation layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
# from tensorflow.python.util.tf_export import keras_export
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.layers.core import Dense
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.framework import ops
from tensorflow.python.ops import standard_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import embedding_ops

def _l2normalizer(v, epsilon=1e-12):
    return v / (K.sum(v ** 2) ** 0.5 + epsilon)


def power_iteration(W, u, rounds=1):
    '''
    Accroding the paper, we only need to do power iteration one time.
    '''
    _u = u

    for i in range(rounds):
        _v = _l2normalizer(K.dot(_u, K.transpose(W)))
        _u = _l2normalizer(K.dot(_v, W))

    W_sn = K.squeeze(K.dot(K.dot(_v, W), K.transpose(_u)), axis=-1)
    return W_sn, _u, _v

# @keras_export('keras.layers.Conv2DSN', 'keras.layers.Convolution2DSN')
class Conv2DSN(Conv2D):
    """2D convolution layer (e.g. spatial convolution over images).

    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.

    Arguments:
      filters: Integer, the dimensionality of the output space
        (i.e. the number of output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string,
        one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape
        `(batch, channels, height, width)`.
        It defaults to the `image_data_format` value found in your
        Keras config file at `~/.keras/keras.json`.
        If you never set it, then it will be "channels_last".
      dilation_rate: an integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix.
      bias_initializer: Initializer for the bias vector.
      kernel_regularizer: Regularizer function applied to
        the `kernel` weights matrix.
      bias_regularizer: Regularizer function applied to the bias vector.
      activity_regularizer: Regularizer function applied to
        the output of the layer (its "activation")..
      kernel_constraint: Constraint function applied to the kernel matrix.
      bias_constraint: Constraint function applied to the bias vector.

    Input shape:
      4D tensor with shape:
      `(samples, channels, rows, cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, rows, cols, channels)` if data_format='channels_last'.

    Output shape:
      4D tensor with shape:
      `(samples, filters, new_rows, new_cols)` if data_format='channels_first'
      or 4D tensor with shape:
      `(samples, new_rows, new_cols, filters)` if data_format='channels_last'.
      `rows` and `cols` values might have changed due to padding.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 spectral_normalization=True,
                 **kwargs):
        super(Conv2DSN, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            activity_regularizer=regularizers.get(activity_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)
        self.u = None
        self.spectral_normalization = spectral_normalization

    def compute_spectral_normal(self, weights, training=True):
      # Spectrally Normalized Weight
      if self.spectral_normalization:

        W_shape = weights.shape.as_list()

        if self.u is None:
          # =[number_classes, embedding_size]
          self.u = self.add_weight(
            'sn_estimate',
            shape=[1, W_shape[-1]],
            initializer='normal',
            dtype=weights.dtype,
            trainable=False)

        # Flatten the Tensor
        W_mat = K.reshape(weights, [-1, W_shape[-1]])  # [-1, output_channel]

        W_sn, u, v = power_iteration(W_mat, self.u)

        if training == True:
          # Update estimated 1st singular vector
          self.u.assign(u)

        W_mat = W_mat / W_sn
        w_bar = K.reshape(W_mat, W_shape)

        return w_bar
      else:
        return weights


    def call(self, inputs, training=True):

        outputs = K.conv2d(
            inputs,
            self.compute_spectral_normal(self.kernel, training),
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if self.bias is not None:
            outputs = K.bias_add(
                outputs,
                self.bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

# @keras_export('keras.layers.DenseSN')
class DenseSN(Dense):
    """Just your regular densely-connected NN layer.
    `Dense` implements the operation:
    `output = activation(dot(input, kernel) + bias)`
    where `activation` is the element-wise activation function
    passed as the `activation` argument, `kernel` is a weights matrix
    created by the layer, and `bias` is a bias vector created by the layer
    (only applicable if `use_bias` is `True`).
    Note: if the input to the layer has a rank greater than 2, then
    it is flattened prior to the initial dot product with `kernel`.
    Example:
    ```python
        # as first layer in a sequential model:
        model = Sequential()
        model.add(Dense(32, input_shape=(16,)))
        # now the model will take as input arrays of shape (*, 16)
        # and output arrays of shape (*, 32)
        # after the first layer, you don't need to specify
        # the size of the input anymore:
        model.add(Dense(32))
    ```
    Arguments:
        units: Positive integer, dimensionality of the output space.
        activation: Activation function to use.
            If you don't specify anything, no activation is applied
            (ie. "linear" activation: `a(x) = x`).
        use_bias: Boolean, whether the layer uses a bias vector.
        kernel_initializer: Initializer for the `kernel` weights matrix.
        bias_initializer: Initializer for the bias vector.
        kernel_regularizer: Regularizer function applied to
            the `kernel` weights matrix.
        bias_regularizer: Regularizer function applied to the bias vector.
        activity_regularizer: Regularizer function applied to
            the output of the layer (its "activation")..
        kernel_constraint: Constraint function applied to
            the `kernel` weights matrix.
        bias_constraint: Constraint function applied to the bias vector.
    Input shape:
        nD tensor with shape: `(batch_size, ..., input_dim)`.
        The most common situation would be
        a 2D input with shape `(batch_size, input_dim)`.
    Output shape:
        nD tensor with shape: `(batch_size, ..., units)`.
        For instance, for a 2D input with shape `(batch_size, input_dim)`,
        the output would have shape `(batch_size, units)`.
    """

    def __init__(self,
                 units,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 spectral_normalization=True,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(DenseSN, self).__init__(
            activity_regularizer=regularizers.get(activity_regularizer),
            units=int(units),
            activation=activations.get(activation),
            use_bias=use_bias,
            kernel_initializer=initializers.get(kernel_initializer),
            bias_initializer=initializers.get(bias_initializer),
            kernel_regularizer=regularizers.get(kernel_regularizer),
            bias_regularizer=regularizers.get(bias_regularizer),
            kernel_constraint=constraints.get(kernel_constraint),
            bias_constraint=constraints.get(bias_constraint),
            **kwargs)

        self.u = None
        self.spectral_normalization = spectral_normalization

    def compute_spectral_normal(self, weights, training=True):
      if self.spectral_normalization:
        W_shape = weights.shape.as_list()

        if self.u is None:
          # =[number_classes, embedding_size]
          self.u = self.add_weight(
            'sn_estimate',
            shape=[1, W_shape[-1]],
            initializer='normal',
            dtype=weights.dtype,
            trainable=False)

        # Flatten the Tensor
        W_mat = K.reshape(weights, [-1, W_shape[-1]])  # [-1, output_channel]

        W_sn, u, v = power_iteration(W_mat, self.u)

        if training == True:
          # Update estimated 1st singular vector
          self.u.assign(u)

        W_mat = W_mat / W_sn
        w_bar = K.reshape(W_mat, W_shape)

        return w_bar
      else:
        return weights

    def call(self, inputs, training=True):

        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        shape = inputs.get_shape().as_list()
        if len(shape) > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, self.compute_spectral_normal(self.kernel, training), [[len(shape) - 1],
                                                                   [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, self.compute_spectral_normal(self.kernel, training))
        if self.use_bias:
            outputs = nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs

# @keras_export('keras.layers.EmbeddingSN')
class EmbeddingSN(Embedding):
  """Turns positive integers (indexes) into dense vectors of fixed size.
  e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`
  This layer can only be used as the first layer in a model.
  Example:
  ```python
  model = Sequential()
  model.add(Embedding(1000, 64, input_length=10))
  # the model will take as input an integer matrix of size (batch,
  # input_length).
  # the largest integer (i.e. word index) in the input should be no larger
  # than 999 (vocabulary size).
  # now model.output_shape == (None, 10, 64), where None is the batch
  # dimension.
  input_array = np.random.randint(1000, size=(32, 10))
  model.compile('rmsprop', 'mse')
  output_array = model.predict(input_array)
  assert output_array.shape == (32, 10, 64)
  ```
  Arguments:
    input_dim: int > 0. Size of the vocabulary,
      i.e. maximum integer index + 1.
    output_dim: int >= 0. Dimension of the dense embedding.
    embeddings_initializer: Initializer for the `embeddings` matrix.
    embeddings_regularizer: Regularizer function applied to
      the `embeddings` matrix.
    embeddings_constraint: Constraint function applied to
      the `embeddings` matrix.
    mask_zero: Whether or not the input value 0 is a special "padding"
      value that should be masked out.
      This is useful when using recurrent layers
      which may take variable length input.
      If this is `True` then all subsequent layers
      in the model need to support masking or an exception will be raised.
      If mask_zero is set to True, as a consequence, index 0 cannot be
      used in the vocabulary (input_dim should equal size of
      vocabulary + 1).
    input_length: Length of input sequences, when it is constant.
      This argument is required if you are going to connect
      `Flatten` then `Dense` layers upstream
      (without it, the shape of the dense outputs cannot be computed).
  Input shape:
    2D tensor with shape: `(batch_size, input_length)`.
  Output shape:
    3D tensor with shape: `(batch_size, input_length, output_dim)`.
  """

  def __init__(self,
               input_dim,
               output_dim,
               embeddings_initializer='uniform',
               embeddings_regularizer=None,
               activity_regularizer=None,
               embeddings_constraint=None,
               mask_zero=False,
               input_length=None,
               spectral_normalization=True,
               **kwargs):
    if 'input_shape' not in kwargs:
      if input_length:
        kwargs['input_shape'] = (input_length,)
      else:
        kwargs['input_shape'] = (None,)
    dtype = kwargs.pop('dtype', K.floatx())
    super(Embedding, self).__init__(dtype=dtype, **kwargs)

    self.input_dim = input_dim
    self.output_dim = output_dim
    self.embeddings_initializer = initializers.get(embeddings_initializer)
    self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.embeddings_constraint = constraints.get(embeddings_constraint)
    self.mask_zero = mask_zero
    self.supports_masking = mask_zero
    self.input_length = input_length
    self.spectral_normalization = spectral_normalization
    self.u = None

  def compute_spectral_normal(self, weights, training=True):
      # Spectrally Normalized Weight
      if self.spectral_normalization:

        W_shape = weights.shape.as_list()

        if self.u is None:
          # =[number_classes, embedding_size]
          self.u = self.add_weight(
            'sn_estimate',
            shape=[1, W_shape[-1]],
            initializer='normal',
            dtype=weights.dtype,
            trainable=False)

        # Flatten the Tensor
        W_mat = K.reshape(weights, [-1, W_shape[-1]])  # [-1, output_channel]

        W_sn, u, v = power_iteration(W_mat, self.u)

        if training == True:
            # Update estimated 1st singular vector
            self.u.assign(u)

        W_mat = W_mat / W_sn
        w_bar = K.reshape(W_mat, W_shape)

        return w_bar
      else:
        return weights

  def call(self, inputs, training=True):

    dtype = K.dtype(inputs)
    if dtype != 'int32' and dtype != 'int64':
      inputs = math_ops.cast(inputs, 'int32')

    embedding_transpose = self.compute_spectral_normal(K.transpose(self.embeddings), training=True)
    embedding_map_bar = K.transpose(embedding_transpose)
    out = embedding_ops.embedding_lookup(embedding_map_bar,inputs)
    return out

  def get_config(self):
    config = {
        'input_dim': self.input_dim,
        'output_dim': self.output_dim,
        'embeddings_initializer':
            initializers.serialize(self.embeddings_initializer),
        'embeddings_regularizer':
            regularizers.serialize(self.embeddings_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'embeddings_constraint':
            constraints.serialize(self.embeddings_constraint),
        'mask_zero': self.mask_zero,
        'input_length': self.input_length
    }
    base_config = super(Embedding, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# @keras_export('keras.layers.Conv2DTranspose',
#               'keras.layers.Convolution2DTranspose')
class Conv2DTransposeSN(Conv2D):
  """Transposed convolution layer (sometimes called Deconvolution).
  The need for transposed convolutions generally arises
  from the desire to use a transformation going in the opposite direction
  of a normal convolution, i.e., from something that has the shape of the
  output of some convolution to something that has the shape of its input
  while maintaining a connectivity pattern that is compatible with
  said convolution.
  When using this layer as the first layer in a model,
  provide the keyword argument `input_shape`
  (tuple of integers, does not include the sample axis),
  e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
  in `data_format="channels_last"`.
  Arguments:
    filters: Integer, the dimensionality of the output space
      (i.e. the number of output filters in the convolution).
    kernel_size: An integer or tuple/list of 2 integers, specifying the
      height and width of the 2D convolution window.
      Can be a single integer to specify the same value for
      all spatial dimensions.
    strides: An integer or tuple/list of 2 integers,
      specifying the strides of the convolution along the height and width.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Specifying any stride value != 1 is incompatible with specifying
      any `dilation_rate` value != 1.
    padding: one of `"valid"` or `"same"` (case-insensitive).
    output_padding: An integer or tuple/list of 2 integers,
      specifying the amount of padding along the height and width
      of the output tensor.
      Can be a single integer to specify the same value for all
      spatial dimensions.
      The amount of output padding along a given dimension must be
      lower than the stride along that same dimension.
      If set to `None` (default), the output shape is inferred.
    data_format: A string,
      one of `channels_last` (default) or `channels_first`.
      The ordering of the dimensions in the inputs.
      `channels_last` corresponds to inputs with shape
      `(batch, height, width, channels)` while `channels_first`
      corresponds to inputs with shape
      `(batch, channels, height, width)`.
      It defaults to the `image_data_format` value found in your
      Keras config file at `~/.keras/keras.json`.
      If you never set it, then it will be "channels_last".
    dilation_rate: an integer or tuple/list of 2 integers, specifying
      the dilation rate to use for dilated convolution.
      Can be a single integer to specify the same value for
      all spatial dimensions.
      Currently, specifying any `dilation_rate` value != 1 is
      incompatible with specifying any stride value != 1.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to the kernel matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    4D tensor with shape:
    `(batch, channels, rows, cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch, rows, cols, channels)` if data_format='channels_last'.
  Output shape:
    4D tensor with shape:
    `(batch, filters, new_rows, new_cols)` if data_format='channels_first'
    or 4D tensor with shape:
    `(batch, new_rows, new_cols, filters)` if data_format='channels_last'.
    `rows` and `cols` values might have changed due to padding.
  References:
    - [A guide to convolution arithmetic for deep
      learning](https://arxiv.org/abs/1603.07285v1)
    - [Deconvolutional
      Networks](https://www.matthewzeiler.com/mattzeiler/deconvolutionalnetworks.pdf)
  """

  def __init__(self,
               filters,
               kernel_size,
               strides=(1, 1),
               padding='valid',
               output_padding=None,
               data_format=None,
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               spectral_normalization=True,
               **kwargs):
    super(Conv2DTransposeSN, self).__init__(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
        activation=activations.get(activation),
        use_bias=use_bias,
        kernel_initializer=initializers.get(kernel_initializer),
        bias_initializer=initializers.get(bias_initializer),
        kernel_regularizer=regularizers.get(kernel_regularizer),
        bias_regularizer=regularizers.get(bias_regularizer),
        activity_regularizer=regularizers.get(activity_regularizer),
        kernel_constraint=constraints.get(kernel_constraint),
        bias_constraint=constraints.get(bias_constraint),
        **kwargs)
    self.spectral_normalization = spectral_normalization
    self.u = None
    self.output_padding = output_padding
    if self.output_padding is not None:
      self.output_padding = conv_utils.normalize_tuple(
          self.output_padding, 2, 'output_padding')
      for stride, out_pad in zip(self.strides, self.output_padding):
        if out_pad >= stride:
          raise ValueError('Stride ' + str(self.strides) + ' must be '
                           'greater than output padding ' +
                           str(self.output_padding))

  def build(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    if len(input_shape) != 4:
      raise ValueError('Inputs should have rank 4. Received input shape: ' +
                       str(input_shape))
    if self.data_format == 'channels_first':
      channel_axis = 1
    else:
      channel_axis = -1
    if input_shape.dims[channel_axis].value is None:
      raise ValueError('The channel dimension of the inputs '
                       'should be defined. Found `None`.')
    input_dim = int(input_shape[channel_axis])
    self.input_spec = InputSpec(ndim=4, axes={channel_axis: input_dim})
    kernel_shape = self.kernel_size + (self.filters, input_dim)

    self.kernel = self.add_weight(
        name='kernel',
        shape=kernel_shape,
        initializer=self.kernel_initializer,
        regularizer=self.kernel_regularizer,
        constraint=self.kernel_constraint,
        trainable=True,
        dtype=self.dtype)
    if self.use_bias:
      self.bias = self.add_weight(
          name='bias',
          shape=(self.filters,),
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          trainable=True,
          dtype=self.dtype)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, training=True):
    inputs_shape = array_ops.shape(inputs)
    batch_size = inputs_shape[0]
    if self.data_format == 'channels_first':
      h_axis, w_axis = 2, 3
    else:
      h_axis, w_axis = 1, 2

    height, width = inputs_shape[h_axis], inputs_shape[w_axis]
    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    # Infer the dynamic output shape:
    out_height = conv_utils.deconv_output_length(height,
                                                 kernel_h,
                                                 padding=self.padding,
                                                 output_padding=out_pad_h,
                                                 stride=stride_h,
                                                 dilation=self.dilation_rate[0])
    out_width = conv_utils.deconv_output_length(width,
                                                kernel_w,
                                                padding=self.padding,
                                                output_padding=out_pad_w,
                                                stride=stride_w,
                                                dilation=self.dilation_rate[1])
    if self.data_format == 'channels_first':
      output_shape = (batch_size, self.filters, out_height, out_width)
    else:
      output_shape = (batch_size, out_height, out_width, self.filters)

    output_shape_tensor = array_ops.stack(output_shape)
    outputs = backend.conv2d_transpose(
        inputs,
        self.compute_spectral_normal(self.kernel, training=training),
        output_shape_tensor,
        strides=self.strides,
        padding=self.padding,
        data_format=self.data_format,
        dilation_rate=self.dilation_rate)

    if not context.executing_eagerly():
      # Infer the static output shape:
      out_shape = self.compute_output_shape(inputs.shape)
      outputs.set_shape(out_shape)

    if self.use_bias:
      outputs = nn.bias_add(
          outputs,
          self.bias,
          data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

    if self.activation is not None:
      return self.activation(outputs)
    return outputs

  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape).as_list()
    output_shape = list(input_shape)
    if self.data_format == 'channels_first':
      c_axis, h_axis, w_axis = 1, 2, 3
    else:
      c_axis, h_axis, w_axis = 3, 1, 2

    kernel_h, kernel_w = self.kernel_size
    stride_h, stride_w = self.strides

    if self.output_padding is None:
      out_pad_h = out_pad_w = None
    else:
      out_pad_h, out_pad_w = self.output_padding

    output_shape[c_axis] = self.filters
    output_shape[h_axis] = conv_utils.deconv_output_length(
        output_shape[h_axis],
        kernel_h,
        padding=self.padding,
        output_padding=out_pad_h,
        stride=stride_h,
        dilation=self.dilation_rate[0])
    output_shape[w_axis] = conv_utils.deconv_output_length(
        output_shape[w_axis],
        kernel_w,
        padding=self.padding,
        output_padding=out_pad_w,
        stride=stride_w,
        dilation=self.dilation_rate[1])
    return tensor_shape.TensorShape(output_shape)

  def get_config(self):
    config = super(Conv2DTransposeSN, self).get_config()
    config['output_padding'] = self.output_padding
    return config


  def compute_spectral_normal(self, weights, training=True):
    # Spectrally Normalized Weight
    if self.spectral_normalization:

      W_shape = weights.shape.as_list()

      if self.u is None:
        # =[number_classes, embedding_size]
        self.u = self.add_weight(
          'sn_estimate',
          shape=[1, W_shape[-1]],
          initializer='normal',
          dtype=weights.dtype,
          trainable=False)

      # Flatten the Tensor
      W_mat = K.reshape(weights, [-1, W_shape[-1]])  # [-1, output_channel]

      W_sn, u, v = power_iteration(W_mat, self.u)

      if training == True:
        # Update estimated 1st singular vector
        self.u.assign(u)

      W_mat = W_mat / W_sn
      w_bar = K.reshape(W_mat, W_shape)

      return w_bar
    else:
      return weights


# @tf_export('keras.layers.CondBatchNormalizationV2', v1=[])
class CondBatchNormalization(BatchNormalization):
  """Batch normalization layer (Ioffe and Szegedy, 2014).

  Normalize the activations of the previous layer at each batch,
  i.e. applies a transformation that maintains the mean activation
  close to 0 and the activation standard deviation close to 1.

  Arguments:
    axis: Integer, the axis that should be normalized
        (typically the features axis).
        For instance, after a `Conv2D` layer with
        `data_format="channels_first"`,
        set `axis=1` in `BatchNormalization`.
    momentum: Momentum for the moving average.
    epsilon: Small float added to variance to avoid dividing by zero.
    center: If True, add offset of `beta` to normalized tensor.
        If False, `beta` is ignored.
    scale: If True, multiply by `gamma`.
        If False, `gamma` is not used.
        When the next layer is linear (also e.g. `nn.relu`),
        this can be disabled since the scaling
        will be done by the next layer.
    beta_initializer: Initializer for the beta weight.
    gamma_initializer: Initializer for the gamma weight.
    moving_mean_initializer: Initializer for the moving mean.
    moving_variance_initializer: Initializer for the moving variance.
    beta_regularizer: Optional regularizer for the beta weight.
    gamma_regularizer: Optional regularizer for the gamma weight.
    beta_constraint: Optional constraint for the beta weight.
    gamma_constraint: Optional constraint for the gamma weight.
    renorm: Whether to use Batch Renormalization
      (https://arxiv.org/abs/1702.03275). This adds extra variables during
      training. The inference is the same for either value of this parameter.
    renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
      scalar `Tensors` used to clip the renorm correction. The correction
      `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
      `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
      dmax are set to inf, 0, inf, respectively.
    renorm_momentum: Momentum used to update the moving means and standard
      deviations with renorm. Unlike `momentum`, this affects training
      and should be neither too small (which would add noise) nor too large
      (which would give stale estimates). Note that `momentum` is still applied
      to get the means and variances for inference.
    fused: if `True`, use a faster, fused implementation, or raise a ValueError
      if the fused implementation cannot be used. If `None`, use the faster
      implementation if possible. If False, do not used the fused
      implementation.
    trainable: Boolean, if `True` also add variables to the graph collection
      `GraphKeys.TRAINABLE_VARIABLES` (see tf.Variable).
    virtual_batch_size: An `int`. By default, `virtual_batch_size` is `None`,
      which means batch normalization is performed across the whole batch. When
      `virtual_batch_size` is not `None`, instead perform "Ghost Batch
      Normalization", which creates virtual sub-batches which are each
      normalized separately (with shared gamma, beta, and moving statistics).
      Must divide the actual batch size during execution.
    adjustment: A function taking the `Tensor` containing the (dynamic) shape of
      the input tensor and returning a pair (scale, bias) to apply to the
      normalized values (before gamma and beta), only during training. For
      example, if axis==-1,
        `adjustment = lambda shape: (
          tf.random_uniform(shape[-1:], 0.93, 1.07),
          tf.random_uniform(shape[-1:], -0.1, 0.1))`
      will scale the normalized value by up to 7% up or down, then shift the
      result by up to 0.1 (with independent scaling and bias for each feature
      but shared across all examples), and finally apply gamma and/or beta. If
      `None`, no adjustment is applied. Cannot be specified if
      virtual_batch_size is specified.

  Input shape:
      Arbitrary. Use the keyword argument `input_shape`
      (tuple of integers, does not include the samples axis)
      when using this layer as the first layer in a model.

  Output shape:
      Same shape as input.

  References:
      - [Batch Normalization: Accelerating Deep Network Training by Reducing
        Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
  """

  # The BatchNormalizationV1 subclass sets this to False to use the V1 behavior.
  _USE_V2_BEHAVIOR = True

  def __init__(self,
               axis=-1,
               momentum=0.99,
               epsilon=1e-3,
               center=True,
               scale=True,
               beta_initializer='zeros',
               gamma_initializer='ones',
               moving_mean_initializer='zeros',
               moving_variance_initializer='ones',
               beta_regularizer=None,
               gamma_regularizer=None,
               beta_constraint=None,
               gamma_constraint=None,
               renorm=False,
               renorm_clipping=None,
               renorm_momentum=0.99,
               fused=None,
               trainable=True,
               virtual_batch_size=None,
               adjustment=None,
               name=None,
               **kwargs):
    super(CondBatchNormalization, self).__init__(
        name=name, trainable=trainable, **kwargs)
    if isinstance(axis, list):
      self.axis = axis[:]
    elif isinstance(axis, int):
      self.axis = axis
    else:
      raise TypeError('axis must be int or list, type given: %s'
                      % type(self.axis))
    self.momentum = momentum
    self.epsilon = epsilon
    self.center = center
    self.scale = scale
    self.beta_initializer = initializers.get(beta_initializer)
    self.gamma_initializer = initializers.get(gamma_initializer)
    self.moving_mean_initializer = initializers.get(moving_mean_initializer)
    self.moving_variance_initializer = initializers.get(
        moving_variance_initializer)
    self.beta_regularizer = regularizers.get(beta_regularizer)
    self.gamma_regularizer = regularizers.get(gamma_regularizer)
    self.beta_constraint = constraints.get(beta_constraint)
    self.gamma_constraint = constraints.get(gamma_constraint)
    self.renorm = renorm
    self.virtual_batch_size = virtual_batch_size
    self.adjustment = adjustment
    if self._USE_V2_BEHAVIOR:
      if fused:
        self._raise_if_fused_cannot_be_used()
      # We leave fused as None if self._fused_can_be_used()==True, since we
      # still may set it to False in self.build() if the input rank is not 4.
      elif fused is None and not self._fused_can_be_used():
        fused = False
    elif fused is None:
      fused = True
    self.supports_masking = True

    self.fused = fused
    self._bessels_correction_test_only = True

    if renorm:
      renorm_clipping = renorm_clipping or {}
      keys = ['rmax', 'rmin', 'dmax']
      if set(renorm_clipping) - set(keys):
        raise ValueError('renorm_clipping %s contains keys not in %s' %
                         (renorm_clipping, keys))
      self.renorm_clipping = renorm_clipping
      self.renorm_momentum = renorm_momentum

  def call(self, inputs, labels, training=None):
    if training is None:
      training = K.learning_phase()

    in_eager_mode = context.executing_eagerly()
    if self.virtual_batch_size is not None:
      # Virtual batches (aka ghost batches) can be simulated by reshaping the
      # Tensor and reusing the existing batch norm implementation
      original_shape = [-1] + inputs.shape.as_list()[1:]
      expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

      # Will cause errors if virtual_batch_size does not divide the batch size
      inputs = array_ops.reshape(inputs, expanded_shape)

      def undo_virtual_batching(outputs):
        outputs = array_ops.reshape(outputs, original_shape)
        return outputs

    if self.fused:
      outputs = self._fused_batch_norm(inputs, training=training)
      if self.virtual_batch_size is not None:
        # Currently never reaches here since fused_batch_norm does not support
        # virtual batching
        outputs = undo_virtual_batching(outputs)
      return outputs

    # Compute the axes along which to reduce the mean / variance
    input_shape = inputs.get_shape()
    ndims = len(input_shape)
    reduction_axes = [i for i in range(ndims) if i not in self.axis]
    if self.virtual_batch_size is not None:
      del reduction_axes[1]     # Do not reduce along virtual batch dim

    # Broadcasting only necessary for single-axis batch norm where the axis is
    # not the last dimension
    broadcast_shape = [1] * ndims
    broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value
    def _broadcast(v):
      if (v is not None and
          len(v.get_shape()) != ndims and
          reduction_axes != list(range(ndims - 1))):
        return array_ops.reshape(v, broadcast_shape)
      return v

    scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

    def _compose_transforms(scale, offset, then_scale, then_offset):
      if then_scale is not None:
        scale *= then_scale
        offset *= then_scale
      if then_offset is not None:
        offset += then_offset
      return (scale, offset)

    # Determine a boolean value for `training`: could be True, False, or None.
    training_value = tf_utils.constant_value(training)
    if training_value is not False:
      if self.adjustment:
        adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
        # Adjust only during training.
        adj_scale = tf_utils.smart_cond(training,
                                        lambda: adj_scale,
                                        lambda: array_ops.ones_like(adj_scale))
        adj_bias = tf_utils.smart_cond(training,
                                       lambda: adj_bias,
                                       lambda: array_ops.zeros_like(adj_bias))
        scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

      # Some of the computations here are not necessary when training==False
      # but not a constant. However, this makes the code simpler.
      keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
      mean, variance = self._moments(
          inputs, reduction_axes, keep_dims=keep_dims)

      moving_mean = self.moving_mean
      moving_variance = self.moving_variance

      mean = tf_utils.smart_cond(training,
                                 lambda: mean,
                                 lambda: moving_mean)
      variance = tf_utils.smart_cond(training,
                                     lambda: variance,
                                     lambda: moving_variance)

      if self.virtual_batch_size is not None:
        # This isn't strictly correct since in ghost batch norm, you are
        # supposed to sequentially update the moving_mean and moving_variance
        # with each sub-batch. However, since the moving statistics are only
        # used during evaluation, it is more efficient to just update in one
        # step and should not make a significant difference in the result.
        new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
        new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
      else:
        new_mean, new_variance = mean, variance

      if self.renorm:
        r, d, new_mean, new_variance = self._renorm_correction_and_moments(
            new_mean, new_variance, training)
        # When training, the normalized values (say, x) will be transformed as
        # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
        # = x * (r * gamma) + (d * gamma + beta) with renorm.
        r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
        d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
        scale, offset = _compose_transforms(r, d, scale, offset)

      def _do_update(var, value):
        if in_eager_mode and not self.trainable:
          return

        return self._assign_moving_average(var, value, self.momentum)

      mean_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_mean, new_mean),
          lambda: self.moving_mean)
      variance_update = tf_utils.smart_cond(
          training,
          lambda: _do_update(self.moving_variance, new_variance),
          lambda: self.moving_variance)
      if not context.executing_eagerly():
        self.add_update(mean_update, inputs=True)
        self.add_update(variance_update, inputs=True)

    else:
      mean, variance = self.moving_mean, self.moving_variance

    beta = K.gather(offset, labels)
    beta = K.expand_dims(K.expand_dims(beta, 1), 1)

    gamma = K.gather(scale, labels)
    gamma = K.expand_dims(K.expand_dims(gamma, 1), 1)

    mean = math_ops.cast(mean, inputs.dtype)
    variance = math_ops.cast(variance, inputs.dtype)
    if offset is not None:
      offset = math_ops.cast(offset, inputs.dtype)
    outputs = nn.batch_normalization(inputs,
                                     _broadcast(mean),
                                     _broadcast(variance),
                                     beta,
                                     gamma,
                                     self.epsilon)
    # If some components of the shape got lost due to adjustments, fix that.
    outputs.set_shape(input_shape)

    if self.virtual_batch_size is not None:
      outputs = undo_virtual_batching(outputs)
    return outputs