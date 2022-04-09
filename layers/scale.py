import tensorflow as tf

from tensorflow_addons.layers import InstanceNormalization, GroupNormalization
from tensorflow.keras.layers import LayerNormalization, BatchNormalization

from scipy.signal import resample_poly

from .layer_util import standardize, minmax_scale, median_norm, max_norm
from .layer_util import quantile_norm, robust_scale, replace_nan_inf, log10

class DownsamplePoly(tf.keras.layers.Layer):
    def __init__(self, du=640, su=50):
        super(DownsamplePoly, self).__init__()
        self.du = int(du)
        self.su = int(su)

    def build(self, input_shape):
        self.batched = len(input_shape) >= 3

    def call(self, x):
        if not self.batched:
            out = tf.convert_to_tensor(
                resample_poly(x.numpy(), self.su, self.du), dtype=x.dtype)
        else:
            resample = lambda x: tf.convert_to_tensor(
                resample_poly(x.numpy(), self.su, self.du), dtype=x.dtype)
            out = tf.map_fn(resample, x)
        return tf.convert_to_tensor(out, dtype=x.dtype)

class NaturalLog(tf.keras.layers.Layer):
    def __init__(self, center=True, abs=False, rpl='max'):
        super(NaturalLog, self).__init__()
        self.center = center
        self.abs = abs
        self.rpl = rpl

    def build(self, input_shape):
        pass

    def call(self, x):
        if self.center:
            x = x + tf.abs(tf.reduce_min(x))
        if self.abs:
            x = tf.abs(x)
        return tf.cast(
            replace_nan_inf(tf.math.log(x), self.rpl), dtype=x.dtype)

class LogScale(tf.keras.layers.Layer):
    def __init__(self, center=True, abs=False, replace='max'):
        super(LogScale, self).__init__()
        self.center = center
        self.abs = abs
        self.replace = replace

    def build(self, input_shape):
        pass

    def call(self, x):
        if self.center:
            x = x + tf.abs(tf.reduce_min(x))
        if self.abs:
            x = tf.math.abs(x)
        return replace_nan_inf(tf.math.log(x), rpl=self.replace)

class Log10Scale(tf.keras.layers.Layer):
    def __init__(self, center=True, abs=False, replace='max'):
        super(Log10Scale, self).__init__()
        self.center = center
        self.abs = abs
        self.replace = replace

    def build(self, input_shape):
        pass

    def call(self, x):
        if self.center:
            x = x + tf.abs(tf.reduce_min(x))
        if self.abs:
            x = tf.math.abs(x)
        return replace_nan_inf(log10(x), rpl=self.replace)

class MedianNorm(tf.keras.layers.Layer):
    def __init__(self, center=True, abs=False, replace='median'):
        super(MedianNorm, self).__init__()
        self.center = center
        self.abs = abs
        self.replace = replace

    def build(self, input_shape):
        pass

    def call(self, x):
        if self.center:
            x = x - tf.reduce_min(x)
        if self.abs:
            x = tf.math.abs(x)
        return replace_nan_inf(median_norm(x), rpl=self.replace)

class NormLogDb(tf.keras.layers.Layer):
    def __init__(self, center=False, abs=True, replace='median'):
        super(NormLogDb, self).__init__()
        self.center = center
        self.abs = abs
        self.replace = replace

    def build(self, input_shape):
        pass

    def call(self, x):
        if self.center:
            x = x + tf.abs(tf.reduce_min(x))
        if self.abs:
            x = tf.math.abs(x)
        return replace_nan_inf(10*log10(x), rpl=self.replace)

def noam_norm(x, epsilon=None):
  """One version of layer normalization."""
  if epsilon is None:
    epsilon = 1.0
  shape = x.shape
  ndims = len(shape)
  return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
    tf.cast(shape[-1], 'float32')))

def get_norm(norm_type, epsilon=1e-16):
  """Get Normalization Callable."""
  if norm_type == "instance":
    return InstanceNormalization()
  if norm_type == "layer":
    return LayerNormalization(epsilon=epsilon)
  if norm_type == "group":
    return GroupNormalization(epsilon=epsilon)
  if norm_type == "batch":
    return BatchNormalization(epsilon=epsilon)
  if norm_type == "noam":
    return noam_norm
  if norm_type == "standard":
      return standardize
  if norm_type == "minmax":
      return minmax_scale
  if norm_type == "median":
      return median_norm
  if norm_type == "max":
      return max_norm
  if norm_type == "quantile":
      return quantile_norm
  if norm_type == "robust":
      return robust_scale
  if norm_type == "log":
      return LogScale()
  if norm_type == "log10":
      return Log10Scale()
  if norm_type == "decibel":
      return NormLogDb()
  print("Normalization type {} not understood".format(norm_type))
  print("Returning tf.identity...")
  return tf.identity

class NormalizePoly(tf.keras.layers.Layer):
    def __init__(self, norm_type='l2', center=False, abs=False, replace=None):
        super(NormalizePoly, self).__init__()
        self.center = center
        self.abs = abs
        self.replace = replace
        self.norm = get_norm(norm_type)

    def build(self, input_shape):
        self.batched = len(input_shape) >= 3

    def call(self, x):
        if self.center:
            x = x - tf.reduce_min(x)
        if self.abs:
            x = tf.math.abs(x)
        if self.replace is not None:
            return replace_nan_inf(self.norm(x), rpl=self.replace)
        return self.norm(x)

if False:
    x = tf.random.normal([8, 256])
    nm=NormalizePoly('instance')
    print(nm(x))
    nm=NormalizePoly('layer')
    print(nm(x))
    nm=NormalizePoly('group')
    print(nm(x))
    nm=NormalizePoly('batch')
    print(nm(x))
    nm=NormalizePoly('noam')
    print(nm(x))
    nm=NormalizePoly('standard')
    nm=NormalizePoly('minmax')
    print(nm(x))
    nm=NormalizePoly('median')
    print(nm(x))
    nm=NormalizePoly('max')
    print(nm(x))
    nm=NormalizePoly('quantile')
    print(nm(x))
    nm=NormalizePoly('robust')
    print(nm(x))
    nm=NormalizePoly('log')
    print(nm(x))
    nm=NormalizePoly('log10')
    print(nm(x))
    nm=NormalizePoly('decibel')
    print(nm(x))



