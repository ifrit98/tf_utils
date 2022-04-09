import os
import scipy
import numpy as np
import tensorflow as tf
from pprint import pprint
from tensorflow.python.framework.ops import Tensor, EagerTensor
from tensorflow.python.data.ops.dataset_ops import BatchDataset
from tensorflow.python.framework.config import list_physical_devices


def exists_here(object_str):
    if str(object_str) != object_str:
        print("Warning: Object passed in was not a string, and may have unexpected behvaior")
    return object_str in list(globals())

def restore_model(model_path, init_compile=False, custom_objects=None, compile_kwargs=None):
    """ Load and compile saved tensorflow model objects from disk. It is recommended
        you use method A if possible, as it does not required named layers, metrics, or
        optimizers in order to associate function handles and is inherently less brittle.

    params: 
        model_path: str path to directory that contains .pb model file
        init_compile: bool whether to attempt to compile the model 'as-is' on load
        custom_objects: dict of object_name, object_handle pairs required to load custom model
        compile_kwargs: dict of kwargs to call compile() on the resultant (custom) model object.
    returns:
        tensorflow model object

    path = './models/aleatoric'

    # Method A - using `compile_kwargs` dict to compile after initial load
    args = {
        'optimizer': 'adam',
        'loss': {'logits_variance': bayesian_categorical_crossentropy(100, 1),
        'sigmoid_output': binary_weighted_crossentropy(1.5)}, 
        'metrics': {},
        'loss_weights': {'logits_variance': 0.2, 'sigmoid_output': 1.0}} 
    model = restore_model(path, compile_kwargs=args)

    # Method B - using `custom_objects` dict to compile on initial load
    num_monte_carlo = 100
    loss_weight = 1.5
    custom_objs = {
        'bayesian_categorical_crossentropy_internal': bayesian_categorical_crossentropy(num_monte_carlo, 1),
        'binary_weighted_crossentropy_internal': binary_weighted_crossentropy(loss_weight)}
    model = restore_model(model_path, init_compile=True, custom_objects=custom_objs)

    print(model.summary())
    """
    if not os.path.exists(model_path):
        raise ValueError("No model found at {}".format(model_path))
    try:
        model = tf.keras.models.load_model(
            model_path, compile=init_compile, custom_objects=custom_objects)
        if compile_kwargs is not None:
            model.compile(**compile_kwargs)
    except :
        raise ImportError(
            "Error loading model {}".format(model_path))
    return model


def grab(dataset):
    r"""Convenient but expensive way to quickly view a batch.
        Args:
            dataset: A tensorflow dataset object.
        Returns:
            nb: dict, a single batch of data, having forced evaluation of
            lazy map calls.
    """
    return next(dataset.as_numpy_iterator())

def is_strlike(x):
    if is_tensor(x):
        return x.dtype == tf.string
    if type(x) == bytes:
        return type(x.decode()) == str
    if is_numpy(x):
        try:
            return 'str' in x.astype('str').dtype.name
        except:
            return False
    return type(x) == str

def is_bool(x):
    if is_tensor(x):
        return x.dtype == tf.bool
    if x not in [True, False, 0, 1]:
        return False
    return True

# TODO: figure out namespace management.  How do get locals injected?
def stopifnot(predicate):
    """
    Evaluate a predicate, quit and display error message if False.

    Params:
        predicate: str containing well-formed python syntax which is to be 
            executed by `eval()` to return a boolean.
    Usage:
        x = 10
        y = 12
        stopifnot("x == y")
        >>> SystemExit:
        >>> Predicate:
        >>>  x == y
        >>> is not True... exiting.
    """
    predicate_str = predicate
    if is_strlike(predicate):
        predicate = eval(predicate)
    if is_bool(predicate) and predicate not in [True, 1]:
        import sys
        sys.exit("\nPredicate:\n\n  {}\n\n is not True... exiting.".format(
            predicate_str))

def list_devices():
  return list_physical_devices()

def list_device_names(XLA=False):
    out = list(map(lambda x: x.name, list_devices()))
    if not XLA:
        out = [i for i in out if not ":XLA_" in i]
    return out

def count_gpus_available():
  x = list_device_names()
  return len(x) - 1
  
def tf_counts(arr, x):
    arr = tf.constant(arr)
    return tf.where(arr == x).shape[0]

def is_numpy(x):
    return x.__class__ in [
        np.ndarray,
        np.rec.recarray,
        np.char.chararray,
        np.ma.masked_array
    ]

def is_tensor(x):
    return x.__class__ in [Tensor, EagerTensor]

def is_complex_tensor(x):
    if not is_tensor(x):
        return False
    return x.dtype.is_complex

def is_float_tensor(x):
    if not is_tensor(x):
        return False
    return x.dtype.is_floating

def is_integer_tensor(x):
    if not is_tensor(x):
        return False
    return x.dtype.is_integer

def as_tensor(x, dtype=None):
    if x is None: return x

    if type(dtype) == str:
        dtype = tf.as_dtype(dtype)

    if is_tensor(x) and not (dtype is None):
        return tf.cast(x, dtype)
    else:
        # this can do an overflow, but it'll issue a warning if it does
        # in that case, use tf$cast() instead of tf$convert_to_tensor, but
        # at that range precision is probably not guaranteed.
        # the right fix then is tf$convert_to_tensor('float64') %>% tf$cast('int64')
        return tf.convert_to_tensor(x, dtype=dtype)

def as_float_tensor(x):
    return as_tensor(x, tf.float32)

def as_double_tensor(x):
    return as_tensor(x, tf.float64)

def as_integer_tensor(x):
    return as_tensor(x, tf.int32)

def as_complex_tensor(x):
    return as_tensor(x, tf.complex64)

def is_empty(tensor):
    if not is_tensor(tensor):
        tensor = as_tensor(tensor)
    return tf.equal(tf.size(tensor), 0)

def as_scalar(x):
    if (len(tf.shape(x))):
        x = tf.squeeze(x)
        try:
            tf.assert_rank(x, 0)
        except:
            raise ValueError("Argument `x` must be of rank <= 1")
    return x

def as_scalar_integer_tensor(x, dtype=tf.int32):
    if dtype not in [tf.int32, tf.int64, 'int32', 'int64']:
        raise ValueError("`dtype` must be integer valued")
    return as_scalar(as_tensor(x, dtype=dtype))

def assert_in_range(freq, lower=0, upper=1):
    tf.compat.v1.assert_greater_equal(freq, lower)
    tf.compat.v1.assert_less_equal(freq, upper)

def assert_is_odd(x):
    k = lambda y: tf.cast(y, x.dtype)
    return tf.assert_equal(x % k(2), k(1))

def assert_length(x, length):
    return tf.assert_equal(
        as_scalar_integer_tensor(length), 
        tf.reduce_prod(x.shape))

def assert_length2(x, valid_lengths):
    length = tf.reduce_prod(tf.shape(x))
    valid_lengths = as_integer_tensor(valid_lengths)
    x = tf.control_dependencies([
            tf.Assert(tf.reduce_any(length == valid_lengths),
            [x])
        ])
    return x

def is_scalar(x):
    if is_tensor(x):
        return x.ndim == 0
    if isinstance(x, str) or type(x) == bytes:
        return True
    if hasattr(x, "__len__"):
        return len(x) == 1
    try:
        x = iter(x)
    except:
        return True
    return np.asarray(x).ndim == 0

def first(x):
    if is_scalar(x):
        return x
    if not is_tensor(x) or is_numpy(x):
        x = as_tensor(x)
    return x[[0] * len(x.shape)]

def last(x):
    if not is_tensor(x) or is_numpy(x):
        x = as_tensor(x)
    return x[[-1] * len(x.shape)]

def safe_rescale(x, epsilon=0.05):
    epsilon = as_float_tensor(epsilon)
    is_complex = is_complex_tensor(x)
    axis = 0 if (is_scalar(x.shape.as_list())) else 1

    if is_complex:
        x = tf.bitcast(x, tf.float32)

    max_vals = tf.reduce_max(x, axis, True)
    max_vals = tf.compat.v2.where(as_float_tensor(max_vals) < epsilon,
                                tf.ones_like(max_vals), max_vals)
    x = x / max_vals

    if is_complex:
        x = tf.bitcast(x, tf.complex64)
    return x

def safe_rescale_graph(signal, axis=0, epsilon=0.05):
    epsilon = as_float_tensor(epsilon)
    is_complex = is_complex_tensor(signal)

    if is_complex:
        signal = tf.bitcast(signal, tf.float32)
    max_vals = tf.reduce_max(signal, axis, True)
    max_vals = tf.compat.v2.where(as_float_tensor(max_vals) < epsilon,
                                tf.ones_like(max_vals), max_vals)
    signal = signal / max_vals

    if is_complex:
        signal = tf.bitcast(signal, tf.complex64)
    return signal

def switch(on, pairs, default=None):
    """ Create dict switch-case from key-word pairs, mimicks R's `switch()`

        Params:
            on: key to index OR predicate returning boolean value to index into dict
            pairs: dict k,v pairs containing predicate enumeration results
        
        Returns: 
            indexed item by `on` in `pairs` dict
        Usage:
        # Predicate
            pairs = {
                True: lambda x: x**2,
                False: lambda x: x // 2
            }
            switch(
                1 == 2, # predicate
                pairs,  # dict 
                default=lambda x: x # identity
            )

        # Index on value
            key = 2
            switch(
                key, 
                values={1:"YAML", 2:"JSON", 3:"CSV"},
                default=0
            )
    """
    if type(pairs) is not dict:
        raise ValueError("`pairs` must be a list of tuple pairs or a dict")
    return pairs.get(on, default)

def types(d, return_dict=False, print_=True):
    r"""Recursively grab dtypes from (nested) dictionary of tensors"""
    types_ = {}
    for k,v in d.items():
        if isinstance(v, dict):
            types_.update(types(v))
        else:
            types_[k] = np.asarray(v).dtype.name
    if print_:
        pprint(types_)
    if return_dict:
        return types_

def shapes(x):
    shapes_fun = FUNCS[type(x)]
    return shapes_fun(x)

def shapes_list(l, print_=False):
    r"""Grab shapes from a list of tensors or numpy arrays"""
    shps = []
    for x in l:
        if print_:
            print(np.asarray(x).shape)
        shps.append(np.asarray(x).shape)
    return shps

def shapes_dict(d, print_=False):
    r"""Recursively grab shapes from potentially nested dictionaries"""
    shps = {}
    for k,v in d.items():
        if isinstance(v, dict):
            shps.update(shapes(v))
        else:
            if print_:
                print(k, ":\t", np.asarray(v).shape)
            shps[k] = np.asarray(v).shape
    return shps

def shapes_tuple(tup, return_shapes=False):
    shps = {i: None for i in range(len(tup))}
    for i, t in enumerate(tup):
        shps[i] = np.asarray(t).shape
    print(shps)
    if return_shapes:
        return shps

FUNCS = {
    dict: shapes_dict,
    list: shapes_list,
    tuple: shapes_tuple
}

def info(d, return_dict=False, print_=True):
    r"""Recursively grab shape, dtype, and size from (nested) dictionary of tensors"""
    info_ = {}
    for k,v in d.items():
        if isinstance(v, dict):
            info_.update(info(v))
        else:
            info_[k] = {
                'size': tf.size(np.asarray(v)).numpy(), 
                'shape' :np.asarray(v).shape, 
                'dtype': np.asarray(v).dtype.name
            }
            if print_:
                _v = np.asarray(v)
                print('key   -', k)
                print('dtype -', _v.dtype.name)
                print('size  -', tf.size(v).numpy())
                print('shape -', _v.shape)
                print()
    if return_dict:
        return info_

def stats(x, axis=None, epsilon=1e-7):
    if is_tensor(x):
        x = x.numpy()
    else:
        x = np.asarray(x)
    if np.min(x) < 0:
        _x = x + abs(np.min(x) - epsilon)
    gmn = scipy.stats.gmean(_x, axis=axis)
    hmn = scipy.stats.hmean(_x, axis=axis)
    mode = scipy.stats.mode(x, axis=axis).mode[0]
    mnt2, mnt3, mnt4 = scipy.stats.moment(x, [2,3,4], axis=axis)
    lq, med, uq = scipy.stats.mstats.hdquantiles(x, axis=axis)
    lq, med, uq = np.quantile(x, [0.25, 0.5, 0.75], axis=axis)
    var = scipy.stats.variation(x, axis=axis) # coefficient of variation
    sem = scipy.stats.sem(x, axis=axis) # std error of the means
    res = scipy.stats.describe(x, axis=axis)
    nms = ['nobs          ', 
           'minmax        ', 
           'mean          ', 
           'variance      ', 
           'skewness      ', 
           'kurtosis      ']
    description = dict(zip(nms, list(res)))
    description.update({
        'coeff_of_var  ': var,
        'std_err_means ': sem,
        'lower_quartile': lq,
        'median        ': med,
        'upper_quartile': uq,
        '2nd_moment    ': mnt2,
        '3rd_moment    ': mnt3,
        '4th_moment    ': mnt4,
        'mode          ': mode,
        'geometric_mean': gmn,
        'harmoinc_mean ': hmn
    })
    return description

def maybe_list_up(x):
    if is_tensor(x):
        if len(tf.shape(x)) == 0:
            return [x]
    else:
        if len(np.asarray(x).shape) == 0:
            return [x]
    return x

def complex_range(x):
    R = tf.math.real(x)
    I = tf.math.imag(x)
    return { 
        'real': (tf.reduce_min(R).numpy(), tf.reduce_max(R).numpy()), 
        'imag': (tf.reduce_min(I).numpy(), tf.reduce_max(I).numpy())
    }

def tfrange(x):
    if is_complex_tensor(x):
        return complex_range(x)
    return (tf.reduce_min(x).numpy(), tf.reduce_max(x).numpy())

def normalize_range(x, lo=-1, hi=1):
    _min = tf.reduce_min(x)
    a = x - _min
    b = tf.reduce_max(x) - _min
    c = hi - lo
    return c * (a / b) + lo

def mnist_data(batch=False, prefetch=False):
    import tensorflow_datasets as tfds
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    if batch:
        ds_train = ds_train.batch(128)
    if prefetch:
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if batch:
        ds_test = ds_test.batch(128)
    ds_test = ds_test.cache()
    if prefetch:
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    return (ds_train, ds_test)


def analytic(x):
    x = as_complex_tensor(x)
    shp = tf.shape(x)
    X = tf.signal.fft(x) / tf.cast(shp[0], tf.complex64)
    xf = as_float_tensor(tf.range(0, shp[0]))
    xf = xf - tf.reduce_mean(xf)
    X = X * as_complex_tensor((1 - tf.sign(xf - 0.5)))
    ifft = tf.signal.ifft(x)
    return ifft

def stft_fix(x, win_len, axis=0):
    pre  = tf.zeros([win_len // 2], dtype=x.dtype)
    post = tf.zeros([tf.cast(tf.math.ceil(win_len / 2), 'int32')], x.dtype)
    return tf.concat([pre, x, post], axis=axis)

def np_dctf(sig, lam=80, eps=2):
    """
    Params:
        sig: array of complex valued SOI
        eps: int epsilon
        lam: int symbol rate expressed as number of samples
    """
    sig = np.asarray(sig)
    sig = sig/np.linalg.norm(sig)
    dsig1 = sig.real[:len(sig)-eps]+sig[eps:].imag*1j
    dsig2 = sig[lam:len(sig)-eps].real-sig[lam+eps:].imag*1j
    dsig = np.multiply(dsig1[:len(dsig1)-lam],dsig2)
    return dsig

def dctf(sig, lam=80, eps=2, sig_dim=0):
    """
    Params:
        sig: array of complex valued SOI
        eps: int epsilon
        lam: int symbol rate expressed as number of samples
    """
    if sig.dtype.name not in ['complex64', 'complex128']:
        pass
    sig = as_tensor(sig, 'complex64')
    sig = sig / tf.linalg.normalize(sig)
    dsig1 = sig.real[:len(sig) - eps] + sig[eps:].imag * 1j
    sig_len = tf.shape(sig)[sig_dim]
    tf.math.real(sig)[:]
    tf.math.imag(sig)
    dsig2 = sig[lam:len(sig)-eps].real-sig[lam+eps:].imag*1j
    dsig = np.multiply(dsig1[:len(dsig1)-lam],dsig2)
    return dsig

"""
Baseband: matlab
https://www.mathworks.com/matlabcentral/answers/132132-converting-a-signal-to-baseband-after-resampling

   Nz = length(z);
   Nv = length(v);
   r = Nz/Nv;
   dt_old = t(2) - t(1);
   dt_new = dt_old/r;
   t_old = t;
   t_new = dt_new*(0:Nv-1);

"""

def load_sig(path, return_fs=False):
    import soundfile as sf 
    x, fs = sf.read(path)
    if x.ndim >= 2:
        x = x[:,0] + x[:,1]
    return x, fs

# TODO: expose analytic_function from tf_dataset for ease of access

def plot_dctf(sig, eps=2, cfreq=0, lam=80, grid_size=50, show_grid=True):
    import matplotlib.pyplot as plt
    """
    Params:
        sig: array of complex valued SOI
        eps: int epsilon
        cfreq: int center frequency (assumed to be basebanded and cfreq==0)
        lam: int symbol rate expressed as number of samples
        grid_size: int 
    """
    #figure properties
    fig, ax = plt.subplots(figsize=(20,15))
    sig = sig/np.linalg.norm(sig) #abs(np.sum(np.multiply(sig,np.conjugate(sig))))

    #ACTUAL DCTF CALCULATION
    dsig1 = sig.real[:len(sig)-eps]+sig[eps:].imag*1j
    dsig2 = sig[lam:len(sig)-eps].real-sig[lam+eps:].imag*1j
    dsig = np.multiply(dsig1[:len(dsig1)-lam],dsig2)
    fig_lim = np.max(np.abs(dsig))+.0001

    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    rmin = -fig_lim
    rmax = fig_lim
    imin = -fig_lim
    imax = fig_lim
    for iq in dsig: #each disg samples
        if abs(iq) > .001:
            r_ind = (((iq.real - rmin) / (rmax - rmin)) * grid_size).astype(np.int32)
            i_ind = (((iq.imag - imin) / (imax - imin)) * grid_size).astype(np.int32)
            try:
                grid[r_ind,i_ind] += 1
            except:
                pass

    #plot iq data
    ax = fig.add_subplot(2,1,2)
    plt.plot(sig[0:200].real)
    plt.plot(sig[0:200].imag)
    ax.set_title("IQ data from beginning of analysis sequence")
    
    #plot dctf grid
    ax2 = fig.add_subplot(3,1,3)
    if show_grid:
        plt.imshow(grid,extent=(-fig_lim,fig_lim,-fig_lim,fig_lim),aspect='auto')
    else:
        plt.plot(dsig.real,dsig.imag,'bo')
        plt.xlim(-fig_lim, fig_lim)
        plt.ylim(-fig_lim, fig_lim)
      
    ax2.set_title("DCTF with symbol rate {}".format(lam))
    plt.show()


def tensor_assign_1D_graph(t, x, idx):
    """
    Pseudo 'in place' modification of tensor `t` with value `x` at index `idx`
    Really just constructs new tensor via slicing... slow, and maybe expensive.
    Param:
        t: tensor to update
        x: value to update (scalar)
        idx: index across `axis` to place `x` (scalar)
    Returns:
        ~t: new tensor with `x` updated at t.shape[axis] = idx
    """
    if not is_tensor(idx):
        idx = tf.constant(idx)
    if len(tf.shape(idx)) == tf.constant(0):
        idx = tf.expand_dims(idx, 0)
    tshape = tf.shape(t)
    hi = tshape - tf.constant(1)
    left = idx
    right = hi - idx
    _left = tf.slice(t, [0], left)
    _right = tf.slice(t, [0], right)
    _x = tf.constant(x, shape=[1], dtype=t.dtype)
    return tf.concat([_left, _x, _right], 0)

def tensor_assign(t, x, idx, axis=-1):
    """
    Pseudo 'in place' modification of tensor `t` with value `x` at index `idx`
    Really just constructs new tensor via slicing... slow, and maybe expensive.
    Param:
        t: tensor to update
        x: value to update (scalar)
        idx: index across `axis` to place `x` (scalar)
    Returns:
        ~t: new tensor with `x` updated at t.shape[axis] = idx
    """
    axis = tf.math.argmax(t.shape) if axis is None else axis
    print("axis:", axis)
    ndim = len(t.shape)
    if idx < 0: 
        raise ValueError("`idx` must be positive.")
    if hasattr(x, '__len__'):
        raise ValueError(
            "`idx` must be a scalar. Currently only support single index.")
    if idx >= t.shape[axis]:
        raise ValueError("`idx` must be <= t.shape[axis]")
    hi   = t.shape[axis] - 1
    left = t.shape.as_list()
    left[axis] = idx
    right = [l for l in left]
    right[axis] = hi-idx
    _mid   = tf.ones([1 for _ in range(len(t.shape))], dtype=t.dtype)
    _left  = tf.slice(t, [0 for _ in range(ndim)], left, t.dtype)
    _right = tf.slice(t, [0 for _ in range(ndim)], right, t.dtype)
    _x = tf.constant(x, shape=_mid.shape, dtype=t.dtype)
    return tf.concat([_left, _x, _right], axis)


def log10(x):
    if 'int' in x.dtype.name:
        x = tf.cast(x, 'float32')
    absolute = tf.math.abs(x)
    n = tf.math.log(absolute)
    d = tf.math.log(tf.constant(10, dtype=n.dtype))
    return n / d