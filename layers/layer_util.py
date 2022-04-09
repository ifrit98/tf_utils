import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from sklearn.preprocessing import normalize, scale, quantile_transform, RobustScaler
from tensorflow.python.framework.ops import Tensor, EagerTensor
permute = tf.keras.backend.permute_dimensions


def is_tensor(x):
    return x.__class__ in [Tensor, EagerTensor]

def as_tensor(x, dtype=None):
    if x is None: 
        return x
    if type(dtype) == str:
        dtype = tf.as_dtype(dtype)
    if is_tensor(x) and not (dtype is None):
        return tf.cast(x, dtype)
    else:
        return tf.convert_to_tensor(x, dtype=dtype)

def is_strlike(x):
    if is_tensor(x):
        return x.dtype == tf.string
    if type(x) == bytes:
        return type(x.decode()) == str
    try:
        x = np.asarray(x)
        return 'str' in x.astype('str').dtype.name
    except:
        pass
    return type(x) == str

def sk_norm(x, norm='l2'):
    if norm not in ['l1', 'l2', 'max']:
        norm = 'l2'
    # (n_samples, n_features)
    return normalize(x, norm=norm)

def sk_scale(x, with_mean=True, with_std=True):
    return scale(x, with_mean=with_mean, with_std=with_std)

def max_norm(x, target_abs_max=1.):
    if not is_tensor(x):
        x = as_tensor(x)
    if not (x.dtype.is_complex or x.dtype.is_floating):
        x = as_tensor(x, 'float32')

    max_vals = tf.reduce_max(tf.abs(x), axis=0)
    max_vals = tf.where(max_vals == 0,
                        tf.ones(list(), max_vals.dtype),
                        max_vals)
    scale_factor = target_abs_max / max_vals
    scale_factor = tf.cast(scale_factor, x.dtype)
    return x * scale_factor

def median(v):
    v = tf.reshape(v, [-1])
    l = v.get_shape()[0]
    mid = l//2 + 1
    val = tf.nn.top_k(v, mid).values
    if l % 2 == 1:
        return val[-1]
    else:
        return 0.5 * (val[-1] + val[-2])

# Expects (batch, samples)
def tfp_median(x, reduce_median=True):
    if len(x.shape) >= 2 and not reduce_median:
        percentile = lambda x: tfp.stats.percentile(
            x, 50.0, interpolation='midpoint')
        return tf.map_fn(percentile, x)
    return tfp.stats.percentile(x, 50.0, interpolation='midpoint')

def minmax_scale(x):
    _min = tf.reduce_min(x)
    _max = tf.reduce_max(x)
    C = tf.subtract(_max, _min)
    return tf.divide(tf.subtract(x, _min), C)

# z-score normalize
def standardize(x):
    if len(x.shape) == 3:
        mus = tf.map_fn(lambda x: tf.math.reduce_mean(x), x)
        stds = tf.map_fn(lambda x: tf.math.reduce_std(x), x)
        return permute((permute(x, [2, 1, 0]) - mus) / stds, [2, 1, 0])
    mu = tf.math.reduce_mean(x)
    std = tf.math.reduce_std(x)
    return tf.divide(tf.subtract(x, mu), std)
z_norm = standardize

def median_norm(x):
    x_rank = len(x.shape)
    permute = tf.keras.backend.permute_dimensions
    if x_rank == 1:
        # (samples,)
        x = tf.expand_dims(x, -1)
        scaled = x - tfp_median(x, reduce_median=True)
        return tf.squeeze(scaled)
    if x_rank == 2:
        if x.shape[0] > x.shape[1]:
            # (samples, features)
            med = tfp_median(permute(x, [1, 0]), reduce_median=False)
            scaled = x - med
        else:
            # (features, samples)
            med = tfp_median(x, reduce_median=False)
            scaled = permute(x, [1, 0]) - med
        return permute(scaled, [1, 0])
    if x_rank == 3:
        # (batch, samples, features)
        perm = [2, 1, 0]
        medians = tf.map_fn(lambda x: tfp_median(permute(x, [1,0])), x)
        scaled = permute(tf.subtract(permute(x, perm), medians), perm)
    return x - tfp_median(x, reduce_median=True)

# expects (n_samples, n_features)
# TODO: Implement 4D Code path for (batch, freq, frames, features)
# TODO: 2D and 3D ARE WRONG! (Must compute across each individual sample, nested tf.map_fn())?
def robust_scale(x):
    x_rank = len(x.shape)
    permute = tf.keras.backend.permute_dimensions
    f = lambda a: RobustScaler().fit_transform(a)
    # Unbatched 1D signal
    if x_rank == 1:
        x = tf.expand_dims(x, -1) # (samples, 1)
        return tf.squeeze(tf.py_function(f, inp=[x], Tout='float32'))
    # Potentially unbatched 2D signal features (samples, features)
    if x_rank == 2:
        if x.shape[0] < x.shape[1]:
            x = permute(x, [1, 0])
            return permute(
                tf.squeeze(tf.py_function(f, inp=[x], Tout='float32')), 
                [1, 0])
        return tf.squeeze(tf.py_function(f, inp=[x], Tout='float32'))
    # batched 3D signal (batch, samples, features)
    if x_rank == 3:
        f_batched = lambda b: tf.map_fn(f, b)
        return tf.squeeze(
            tf.py_function(f_batched, inp=[x], Tout=x.dtype.name))

# About 5x slower than robust_scaler
# TODO: Implement 4D Code path for (batch, freq, frames, features)
# TODO: rework so there is a separate tf_quantile_norm, and py_quantile_norm for graph mode.
def quantile_norm(x):
    x_rank = len(x.shape)
    f = lambda a: quantile_transform(a)
    # Unbatched 1D signal
    if x_rank == 1:
        x = tf.expand_dims(x, -1) # (samples, 1)
        return tf.squeeze(tf.py_function(f, inp=[x], Tout='float32'))
    # Potentially unbatched 2D signal features (samples, features)
    if x_rank == 2:
        if x.shape[0] < x.shape[1]:
            x = permute(x, [1, 0])
            return permute(
                tf.squeeze(tf.py_function(f, inp=[x], Tout='float32')), 
                [1, 0])
        return tf.squeeze(tf.py_function(f, inp=[x], Tout='float32'))
    # batched 3D signal (batch, samples, features)
    if x_rank == 3:
        f_batched = lambda b: tf.map_fn(f, b)
        return tf.squeeze(
            tf.py_function(f_batched, inp=[x], Tout=x.dtype.name))

def log_norm(x, center=True, absolute=False, repl_with='median'):
    if repl_with == 'max':
        rpl = tf.reduce_max(x)
    elif repl_with == 'median':
        rpl = tfp_median(x)
    elif not is_strlike(repl_with):
        rpl = tf.constant(repl_with)
    if center:
        x = x - tf.reduce_min(x)
    if absolute:
        x = tf.abs(x)
    return replace_nan_inf(tf.math.log(x), rpl)

def log10(x):
    if not is_tensor(x):
        x = as_tensor(x)
    if 'int' in x.dtype.name:
        x = tf.cast(x, 'float32')
    absolute = tf.math.abs(x)
    n = tf.math.log(absolute)
    d = tf.math.log(tf.constant(10, dtype=n.dtype))
    return n / d

def nrmPwrDb(x, eps=1e-16):
    x_pow = tf.math.pow(tf.abs(x), 2)
    return 10*log10((x_pow/tf.reduce_max(x_pow)) + eps)

def nrmDb(x):
    if 'complex' in x.dtype.name:
        return nrmDb_cplx(x)
    return 10 * log10(x)

def nrmDb_cplx(x, replace=False):
    if replace:
        f = lambda x: replace_nan_inf(nrmDb(x), 'max')
    else:
        f = lambda x: nrmDb(x)
    return apply_in_real_space(x, f)

def apply_in_real_space(x, f):
    if 'complex' not in x.dtype.name:
        raise ValueError("`x` must be a complex valued tensor")
    return tf.complex(real=f(tf.math.real(x)), imag=f(tf.math.imag(x)))
apply_complex = apply_in_real_space

def apply_bitcast(x, f):
    x = cplx1D_to_float2D(x)
    return float2D_to_cplx1D(f(x))

def cplx1D_to_float2D(x):
    if 'complex' not in x.dtype.name:
        raise ValueError('Expected a complex type...')
    dtype = 'float64' if x.dtype.name == 'complex128' else 'float32'    
    return tf.bitcast(x, dtype)

def float2D_to_cplx1D(x):
    if 'float' not in x.dtype.name:
        raise ValueError('Expected a floating type...')
    dtype = 'complex128' if x.dtype.name == 'float64' else 'complex64'
    return tf.bitcast(x, dtype)

def replace_nan(x, rpl=0):
    if 'complex' in x.dtype.name:
        return apply_in_real_space(x, replace_nan)
    return tf.where(
        tf.math.is_nan(x), 
        tf.zeros_like(x, x.dtype) if rpl == 0 else tf.zeros_like(
            x, x.dtype) + tf.cast(rpl, x.dtype), x)

def replace_inf(x, rpl=0):
    if 'complex' in x.dtype.name:
        return apply_in_real_space(x, replace_nan)
    return tf.where(
        tf.math.is_inf(x), 
        tf.zeros_like(x, x.dtype) if rpl == 0 else tf.zeros_like(
            x, x.dtype) + tf.cast(rpl, x.dtype), x)

def replace_nan_inf(x, rpl=0):
    if 'complex' in x.dtype.name:
        if rpl == 'max':
            rpl = apply_in_real_space(x, tf.reduce_max)
        elif rpl == 'median':
            rpl = apply_in_real_space(x, tfp_median)
        elif not is_strlike(rpl):
            rpl = tf.constant(rpl, dtype=x.dtype)
        f = lambda y: replace_nan(replace_inf(y, rpl), rpl)
        return apply_in_real_space(x, f)
    if rpl == 'max':
        rpl = tf.reduce_max(x)
    elif rpl == 'median':
        rpl = tfp_median(x)
    elif not is_strlike(rpl):
        rpl = tf.constant(rpl, dtype=x.dtype)
    return replace_nan(replace_inf(x, rpl), rpl)

def any_nan(x, dtype=None):
    if x.dtype.name == 'complex64':
        dtype = 'float32'
        R = tf.map_fn(lambda y: tf.cast(tf.math.is_nan(y), dtype), tf.math.real(x))
        I = tf.map_fn(lambda y: tf.cast(tf.math.is_nan(y), dtype), tf.math.imag(x))
        return tf.math.reduce_any(
            tf.cast(R, 'bool')) or tf.math.reduce_any(tf.cast(I, 'bool'))
    if x.dtype.name == 'complex128':
        dtype = 'float64'
        R = tf.map_fn(lambda y: tf.cast(tf.math.is_nan(y), dtype), tf.math.real(x))
        I = tf.map_fn(lambda y: tf.cast(tf.math.is_nan(y), dtype), tf.math.imag(x))    
        return tf.math.reduce_any(
            tf.cast(R, 'bool')) or tf.math.reduce_any(tf.cast(I, 'bool'))
    return tf.math.reduce_any(
        tf.cast(tf.map_fn(lambda x: tf.cast(tf.math.is_nan(x), x.dtype), x), 'bool'))

def any_inf(x, dtype=None):
    if x.dtype.name == 'complex64':
        dtype = 'float32'
        R = tf.map_fn(lambda y: tf.cast(tf.math.is_inf(y), dtype), tf.math.real(x))
        I = tf.map_fn(lambda y: tf.cast(tf.math.is_inf(y), dtype), tf.math.imag(x))
        return tf.math.reduce_any(
            tf.cast(R, 'bool')) or tf.math.reduce_any(tf.cast(I, 'bool'))
    if x.dtype.name == 'complex128':
        dtype = 'float64'
        R = tf.map_fn(lambda y: tf.cast(tf.math.is_inf(y), dtype), tf.math.real(x))
        I = tf.map_fn(lambda y: tf.cast(tf.math.is_inf(y), dtype), tf.math.imag(x))    
        return tf.math.reduce_any(
            tf.cast(R, 'bool')) or tf.math.reduce_any(tf.cast(I, 'bool'))
    return tf.math.reduce_any(
        tf.cast(tf.map_fn(lambda x: tf.cast(tf.math.is_inf(x), x.dtype), x), 'bool'))

def any_nan_or_inf(x, dtype=None):
    return any_nan(x, dtype=dtype) or any_inf(x, dtype=dtype)
