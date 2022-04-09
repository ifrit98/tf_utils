import tensorflow as tf
from _utils.tensor import shape_list
from activations.activations import lrelu, hard_sigmoid, saturating_sigmoid
from layers.layers import InstanceNormalization, GroupNormalization, LayerNormalization


def general_conv(x,
                 num_filters=64,
                 filter_size=7,
                 stride=1,
                 stddev=0.02,
                 padding="VALID",
                 name="conv",
                 do_norm="instance",
                 do_relu=True,
                 relufactor=0):
  """Generalized convolution layer."""
  x = tf.keras.layers.Conv2D(
      num_filters,
      filter_size,
      stride,
      padding,
      activation=None,
      kernel_initializer=tf.compat.v1.truncated_normal_initializer(stddev=stddev),
      bias_initializer=tf.constant_initializer(0.0))(x)
  if do_norm == "layer":
    x = LayerNormalization()(x)
  elif do_norm == "instance":
    x = InstanceNormalization()(x)
  elif do_norm == "group":
      x = GroupNormalization()(x)

  if do_relu:
    if relufactor == 0:
      x = tf.nn.relu(x, "relu")
    else:
      x = lrelu(x, leak=relufactor)

  return x


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
  """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
  static_shape = inputs.get_shape()
  if not static_shape or len(static_shape) != 4:
    raise ValueError("Inputs to conv must have statically known rank 4. "
                     "Shape: " + str(static_shape))
  # Add support for left padding.
  if kwargs.get("padding") == "LEFT":
    dilation_rate = (1, 1)
    if "dilation_rate" in kwargs:
      dilation_rate = kwargs["dilation_rate"]
    assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
    height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
    cond_padding = tf.cond(
        tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
        lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
    width_padding = 0 if static_shape[2] == 1 else cond_padding
    padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
    inputs = tf.pad(inputs, padding)
    # Set middle two dimensions to None to prevent convolution from complaining
    inputs.set_shape([static_shape[0], None, None, static_shape[3]])
    kwargs["padding"] = "VALID"

  def conv2d_kernel(kernel_size_arg, name_suffix):
    """Call conv2d but add suffix to name."""
    name = "{}_{}".format(kwargs.get("name", "conv"), name_suffix)
    original_name = kwargs.pop("name", None)
    original_force2d = kwargs.pop("force2d", None)
    result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
    if original_name is not None:
      kwargs["name"] = original_name  # Restore for other calls.
    if original_force2d is not None:
      kwargs["force2d"] = original_force2d
    return result

  return conv2d_kernel(kernel_size, "single")


def conv(inputs, filters, kernel_size, dilation_rate=(1, 1), **kwargs):
  def _conv2d(x, *args, **kwargs):
    return tf.keras.layers.Conv2D(*args, **kwargs)(x)
  return conv_internal(
      _conv2d,
      inputs,
      filters,
      kernel_size,
      dilation_rate=dilation_rate,
      **kwargs)


def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
  return tf.squeeze(
      conv(tf.expand_dims(inputs, 2), filters, (kernel_size, 1),
           dilation_rate=(dilation_rate, 1), **kwargs),
      2)


def separable_conv(inputs, filters, kernel_size, **kwargs):
  def _sep_conv2d(x, *args, **kwargs):
    return tf.keras.layers.SeparableConv2D(*args, **kwargs)(x)
  return conv_internal(_sep_conv2d, inputs, filters, kernel_size, **kwargs)


def subseparable_conv(inputs, filters, kernel_size, **kwargs):
  """Sub-separable convolution. If separability == 0 it's a separable_conv."""

  def conv_fn(inputs, filters, kernel_size, **kwargs):
    """Sub-separable convolution, splits into separability-many blocks."""
    separability = None
    if "separability" in kwargs:
      separability = kwargs.pop("separability")
    if separability:
      parts = []
      abs_sep = separability if separability > 0 else -1 * separability
      for split_idx, split in enumerate(tf.split(inputs, abs_sep, axis=3)):
        if separability > 0:
          parts.append(
              tf.keras.layers.Conv2D(filters // separability, kernel_size,
                              **kwargs)(split))
        else:
          parts.append(
              tf.keras.layers.SeparableConv2D(filters // abs_sep,
                                        kernel_size, **kwargs)(split))
      if separability > 1:
        result = tf.keras.layers.Conv2D(filters, (1, 1))(tf.concat(parts, axis=3))
      elif abs_sep == 1:  # If we have just one block, return it.
        assert len(parts) == 1
        result = parts[0]
      else:
        result = tf.concat(parts, axis=3)
    else:
      result = tf.keras.layers.SeparableConv2D(filters, kernel_size,
                                        **kwargs)(inputs)
    if separability is not None:
      kwargs["separability"] = separability
    return result

  return conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs)



def conv_block_internal(conv_fn,
                        inputs,
                        filters,
                        dilation_rates_and_kernel_sizes,
                        first_relu=True,
                        use_elu=False,
                        separabilities=None,
                        **kwargs):
  """A block of convolutions.
  Args:
    conv_fn: convolution function, e.g. conv or separable_conv.
    inputs: a Tensor
    filters: an Integer
    dilation_rates_and_kernel_sizes: a list of tuples (dilation, (k_w, k_h))
    first_relu: whether to do a relu at start (defaults to True)
    use_elu: whether to use ELUs instead of ReLUs (defaults to False)
    separabilities: list of separability factors (per-layer).
    **kwargs: additional arguments (e.g., pooling)
  Returns:
     a Tensor.
  """

  name = kwargs.pop("name") if "name" in kwargs else None
  mask = kwargs.pop("mask") if "mask" in kwargs else None

  # Usage for normalize_fn kwarg:
  # if not specified, use layer norm
  # if given normalize_fn=None, don't use any normalization
  # if given normalize_fn=norm, use the specified norm function

  use_layer_norm = "normalizer_fn" not in kwargs
  norm = kwargs.pop("normalizer_fn", None)
  use_normalizer_fn = use_layer_norm or norm

  if use_layer_norm:
    norm = lambda x, name: LayerNormalization()(x, name=name)

  cur, counter = inputs, -1
  for dilation_rate, kernel_size in dilation_rates_and_kernel_sizes:
    counter += 1
    if first_relu or counter > 0:
      cur = tf.nn.elu(cur) if use_elu else tf.nn.relu(cur)
    if mask is not None:
      cur *= mask
    if separabilities:
      cur = conv_fn(
          cur,
          filters,
          kernel_size,
          dilation_rate=dilation_rate,
          name="conv_block_%d" % counter,
          use_bias=norm is None,
          separability=separabilities[counter],
          **kwargs)
    else:
      cur = conv_fn(
          cur,
          filters,
          kernel_size,
          dilation_rate=dilation_rate,
          name="conv_block_%d" % counter,
          use_bias=norm is None,
          **kwargs)
    if use_normalizer_fn:
      cur = norm(cur, name="conv_block_norm_%d" % counter)
  return cur


def conv_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """A block of standard 2d convolutions."""
  return conv_block_internal(conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def conv1d_block(inputs, filters, dilation_rates_and_kernel_sizes, **kwargs):
  """A block of standard 1d convolutions."""
  return conv_block_internal(conv1d, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def separable_conv_block(inputs, filters, dilation_rates_and_kernel_sizes,
                         **kwargs):
  """A block of separable convolutions."""
  return conv_block_internal(separable_conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def subseparable_conv_block(inputs, filters, dilation_rates_and_kernel_sizes,
                            **kwargs):
  """A block of separable convolutions."""
  return conv_block_internal(subseparable_conv, inputs, filters,
                             dilation_rates_and_kernel_sizes, **kwargs)


def pool(inputs, window_size, pooling_type, padding, strides=(1, 1)):
    """Pooling (supports "LEFT")."""
    static_shape = inputs.get_shape()
    if not static_shape or len(static_shape) != 4:
        raise ValueError("Inputs to conv must have a statically known rank of 4.")
    # Add support for left padding.
    if padding == "LEFT":
        assert window_size[0] % 2 == 1 and window_size[1] % 2 == 1
        if len(static_shape) == 3:
            width_padding = 2 * (window_size[1] // 2)
            padding_ = [[0, 0], [width_padding, 0], [0, 0]]
        else:
            height_padding = 2 * (window_size[0] // 2)
            cond_padding = tf.cond(
                tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
                lambda: tf.constant(2 * (window_size[1] // 2)))
            width_padding = 0 if static_shape[2] == 1 else cond_padding
            padding_ = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
        inputs = tf.pad(inputs, padding_)
        inputs.set_shape([static_shape[0], None, None, static_shape[3]])
        padding = "VALID"
    return tf.nn.pool(inputs, window_size, pooling_type, padding, strides=strides)


def conv_block_downsample(x,
                          kernel,
                          strides,
                          padding,
                          separability=0,
                          name=None,
                          reuse=None):
  """Implements a downwards-striding conv block, like Xception exit flow."""
  hidden_size = int(x.get_shape()[-1])
  res = conv_block(
      x,
      int(1.25 * hidden_size), [((1, 1), kernel)],
      padding=padding,
      strides=strides,
      name="res_conv")

  x = subseparable_conv_block(
      x,
      hidden_size, [((1, 1), kernel)],
      padding=padding,
      separability=separability,
      name="conv0")
  x = subseparable_conv_block(
      x,
      int(1.25 * hidden_size), [((1, 1), kernel)],
      padding=padding,
      separability=separability,
      name="conv1")
  x = pool(x, kernel, "MAX", padding, strides=strides)

  x += res

  x = subseparable_conv_block(
      x,
      2 * hidden_size, [((1, 1), kernel)],
      first_relu=False,
      padding=padding,
      separability=separability,
      name="conv2")
  x = subseparable_conv_block(
      x,
      int(2.5 * hidden_size), [((1, 1), kernel)],
      padding=padding,
      separability=separability,
      name="conv3")
  return x


def maybe_zero_out_padding(inputs, kernel_size, nonpadding_mask):
  """If necessary, zero out inputs to a conv for padding positions.
  Args:
    inputs: a Tensor with shape [batch, length, ...]
    kernel_size: an integer or pair of integers
    nonpadding_mask: a Tensor with shape [batch, length]
  Returns:
    Tensor of the same shape as inputs.
  """
  if (kernel_size != 1 and kernel_size != (1, 1) and
      nonpadding_mask is not None):
    while nonpadding_mask.get_shape().ndims < inputs.get_shape().ndims:
      nonpadding_mask = tf.expand_dims(nonpadding_mask, -1)
    return inputs * nonpadding_mask

  return inputs


def conv_relu_conv(inputs,
                   filter_size,
                   output_size,
                   first_kernel_size=3,
                   second_kernel_size=3,
                   padding="SAME",
                   nonpadding_mask=None,
                   dropout=0.0,
                   name=None,
                   cache=None,
                   decode_loop_step=None):
  """Hidden layer with RELU activation followed by linear projection.
  Args:
    inputs: A tensor.
    filter_size: An integer.
    output_size: An integer.
    first_kernel_size: An integer.
    second_kernel_size: An integer.
    padding: A string.
    nonpadding_mask: A tensor.
    dropout: A float.
    name: A string.
    cache: A dict, containing Tensors which are the results of previous
        attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU. If it is not None, the function
        will do inplace update for the cache instead of concatenating the
        current result to the cache.
  Returns:
    A Tensor.
  """
  from tensorflow.python.ops import inplace_ops

  inputs = maybe_zero_out_padding(inputs, first_kernel_size, nonpadding_mask)

  if cache:
    if decode_loop_step is None:
      inputs = cache["f"] = tf.concat([cache["f"], inputs], axis=1)
    else:
      # Inplace update is required for inference on TPU.
      # Inplace_ops only supports inplace_update on the first dimension.
      # The performance of current implementation is better than updating
      # the tensor by adding the result of matmul(one_hot,
      # update_in_current_step)
      tmp_f = tf.transpose(cache["f"], perm=[1, 0, 2])
      tmp_f = inplace_ops.alias_inplace_update(
          tmp_f,
          decode_loop_step * tf.shape(inputs)[1],
          tf.transpose(inputs, perm=[1, 0, 2]))
      inputs = cache["f"] = tf.transpose(tmp_f, perm=[1, 0, 2])
    inputs = cache["f"] = inputs[:, -first_kernel_size:, :]

  h = conv1d(
      inputs, filter_size, first_kernel_size, padding=padding, name="conv1")

  if cache:
    h = h[:, -1:, :]

  h = tf.nn.relu(h)
  if dropout != 0.0:
    h = tf.nn.dropout(h, 1.0 - dropout)
  h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
  return conv1d(
      h, output_size, second_kernel_size, padding=padding, name="conv2")



def sepconv_relu_sepconv(inputs,
                         filter_size,
                         output_size,
                         first_kernel_size=(1, 1),
                         second_kernel_size=(1, 1),
                         padding="LEFT",
                         nonpadding_mask=None,
                         dropout=0.0,
                         name=None):
  """Hidden layer with RELU activation followed by linear projection."""
  inputs = maybe_zero_out_padding(inputs, first_kernel_size, nonpadding_mask)
  if inputs.get_shape().ndims == 3:
    is_3d = True
    inputs = tf.expand_dims(inputs, 2)
  else:
    is_3d = False
  h = separable_conv(
      inputs,
      filter_size,
      first_kernel_size,
      activation=tf.nn.relu,
      padding=padding,
      name="conv1")
  if dropout != 0.0:
    h = tf.nn.dropout(h, 1.0 - dropout)
  h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
  ret = separable_conv(
      h, output_size, second_kernel_size, padding=padding, name="conv2")
  if is_3d:
    ret = tf.squeeze(ret, 2)
  return ret


def conv_gru(x,
             kernel_size,
             filters,
             padding="SAME",
             dilation_rate=(1, 1),
             name=None,
             reuse=None):
  """Convolutional GRU in 1 dimension."""

  # Let's make a shorthand for conv call first.
  def do_conv(args, name, bias_start, padding):
    return conv(
        args,
        filters,
        kernel_size,
        padding=padding,
        dilation_rate=dilation_rate,
        bias_initializer=tf.constant_initializer(bias_start),
        name=name)

  # Here comes the GRU gate.
  reset = saturating_sigmoid(do_conv(x, "reset", 1.0, padding))
  gate = saturating_sigmoid(do_conv(x, "gate", 1.0, padding))
  candidate = tf.tanh(do_conv(reset * x, "candidate", 0.0, padding))
  return gate * x + (1 - gate) * candidate
    


def conv_lstm(x,
              kernel_size,
              filters,
              padding="SAME",
              dilation_rate=(1, 1),
              name=None,
              reuse=None):
  """Convolutional LSTM in 1 dimension."""
  gates = conv(
      x,
      4 * filters,
      kernel_size,
      padding=padding,
      dilation_rate=dilation_rate)
  g = tf.split(LayerNormalization()(gates, 4 * filters), 4, axis=3)
  new_cell = tf.sigmoid(g[0]) * x + tf.sigmoid(g[1]) * tf.tanh(g[3])
  return tf.sigmoid(g[2]) * tf.tanh(new_cell)



def diagonal_conv_gru(x,
                      kernel_size,
                      filters,
                      dropout=0.0,
                      name=None,
                      reuse=None):
  """Diagonal Convolutional GRU as in https://arxiv.org/abs/1702.08727."""

  # Let's make a shorthand for conv call first.
  def do_conv(args, name, bias_start):
    return conv(
        args,
        filters,
        kernel_size,
        padding="SAME",
        bias_initializer=tf.constant_initializer(bias_start),
        name=name)

  # Here comes the GRU gate.
  reset, reset_cost = hard_sigmoid(do_conv(x, "reset", 0.5))
  gate, gate_cost = hard_sigmoid(do_conv(x, "gate", 0.7))
  candidate = tf.tanh(do_conv(reset * x, "candidate", 0.0))

  if dropout > 0.0:
    candidate = tf.nn.dropout(candidate, 1.0 - dropout)

  # Diagonal shift.
  shift_filters = filters // 3
  base_filter = ([[0, 1, 0]] * (filters - 2 * shift_filters) +
                  [[1, 0, 0]] * shift_filters + [[0, 0, 1]] * shift_filters)
  shift_filter = tf.constant(tf.transpose(base_filter), dtype=tf.float32)
  shift_filter = tf.expand_dims(tf.expand_dims(shift_filter, 0), 3)
  x_shifted = tf.nn.depthwise_conv2d(
      x, shift_filter, [1, 1, 1, 1], padding="SAME")

  # Return the gated result and cost.
  total_cost_avg = 0.5 * (reset_cost + gate_cost)
  return gate * x_shifted + (1 - gate) * candidate, total_cost_avg

