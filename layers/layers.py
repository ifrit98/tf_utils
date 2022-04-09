import tensorflow as tf

from .pru import PyramidalRecurrentBlock, PRU
from .glu import GatedConvBlock, GLU
from .antirectifier import Antirectifier
from .squeeze_excite import SEResNeXtBottleneck, SqueezeExcite
from .weight_norm import WeightNorm, WeightNormTFP
from .resblock import ResblockBatchnorm1D, ResblockBatchnorm2D, ResblockBatchnormBottleneck1D, ResblockBatchnormBottleneck2D
from .scale import *

from tensorflow_addons.layers import *
from tensorflow.keras.layers import LayerNormalization, BatchNormalization


def apply_norm(x, norm_type, depth, epsilon):
  """Apply Normalization."""
  if norm_type == "instance":
    return InstanceNormalization()(x)
  if norm_type == "layer":
    return LayerNormalization(epsilon=epsilon)(x)
  if norm_type == "group":
    return GroupNormalization(epsilon=epsilon)(x)
  if norm_type == "batch":
    return BatchNormalization(epsilon=epsilon)(x)
  if norm_type == "noam":
    return noam_norm(x, epsilon)
  if norm_type == "l2":
    return tf.norm(x)
  if norm_type == "none" or not norm_type:
    return x
  raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                   "'noam', 'lr', 'none'.")


def zero_add(previous_value, x, name=None, reuse=None):
  """Resnet connection with zero initialization.
  Another type of resnet connection which returns previous_value + gamma * x.
  gamma is a trainable scalar and initialized with zero. It is useful when a
  module is plugged into a trained model and we want to make sure it matches the
  original model's performance.
  Args:
    previous_value:  A tensor.
    x: A tensor.
    name: name of variable scope; defaults to zero_add.
    reuse: reuse scope.
  Returns:
    previous_value + gamma * x.
  """
  # TODO make sure this works as intented...
  gamma = tf.Variable(tf.zeros(shape=()), name="gamma")
  return previous_value + gamma * x


def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         epsilon,
                         name=None,
                         dropout_broadcast_dims=None):
  """Apply a sequence of functions to the input or output of a layer.
  The sequence is specified as a string which may contain the following
  characters:
    a: add previous_value
    n: apply normalization
    d: apply dropout
    z: zero add
  For example, if sequence=="dna", then the output is
    previous_value + normalize(dropout(x))
  Args:
    previous_value: A Tensor, to be added as a residual connection ('a')
    x: A Tensor to be transformed.
    sequence: a string.
    dropout_rate: a float
    norm_type: a string (see apply_norm())
    depth: an integer (size of last dimension of x).
    epsilon: a float (parameter for normalization)
    default_name: a string
    name: a string
    dropout_broadcast_dims:  an optional list of integers less than 3
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
  Returns:
    a Tensor
  """
  if sequence is None:
    return x
  for c in sequence:
    if c == "a":
      x += previous_value
    elif c == "z":
      x = zero_add(previous_value, x)
    elif c == "n":
      x = apply_norm(
          x, norm_type, depth, epsilon)
    else:
      assert c == "d", ("Unknown sequence step %s" % c)
      x = tf.keras.layers.Dropout(dropout_rate)(x)
  return x


def layer_preprocess(layer_input, norm_type, norm_epsilon, sequence, dropout, broadcast_dims=None):
  """Apply layer preprocessing.
  See layer_prepostprocess() for details.
  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:
    layer_preprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon
  Args:
    layer_input: a Tensor
    hparams: a hyperparameters object.
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
  Returns:
    a Tensor
  """
  assert "a" not in sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  assert "z" not in sequence, (
      "No residual connections allowed in hparams.layer_preprocess_sequence")
  return layer_prepostprocess(
      None,
      layer_input,
      sequence=sequence,
      dropout_rate=dropout,
      norm_type=norm_type,
      depth=None,
      epsilon=norm_epsilon,
      dropout_broadcast_dims=broadcast_dims)


def layer_postprocess(layer_input, layer_output, sequence, norm_type, 
                      norm_epsilon, dropout, broadcast_dims=None):
  """Apply layer postprocessing.
  See layer_prepostprocess() for details.
  A hyperparameters object is passed for convenience.  The hyperparameters
  that may be used are:
    layer_postprocess_sequence
    layer_prepostprocess_dropout
    norm_type
    hidden_size
    norm_epsilon
  Args:
    layer_input: a Tensor
    layer_output: a Tensor
    hparams: a hyperparameters object.
  Returns:
    a Tensor
  """
  return layer_prepostprocess(
      layer_input,
      layer_output,
      sequence=sequence,
      dropout_rate=dropout,
      norm_type=norm_type,
      depth=None,
      epsilon=norm_epsilon,
      dropout_broadcast_dims=broadcast_dims)
