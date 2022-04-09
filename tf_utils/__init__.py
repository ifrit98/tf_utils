
from tensorflow_addons import metrics
from tensorflow_addons import optimizers
from tensorflow_addons import seq2seq
from tensorflow_addons import image
from tensorflow_addons import text

from _utils import utils, tensor
from layers import layers
from callbacks import callbacks
from activations import activations
from losses import losses

from _utils.utils import log10, dctf, analytic, mnist_data, tfrange
from _utils.utils import info, shapes, types, switch, first, last
from _utils.utils import is_scalar, as_scalar, is_tensor, as_tensor
from _utils.utils import is_empty, is_numpy, count_gpus_available, restore_model
from _utils.utils import stopifnot, is_strlike, is_bool, grab, exists_here

from layers.layer_util import nrmDb, any_nan, any_inf, replace_nan, replace_inf
from layers.layer_util import any_nan_or_inf, replace_nan_inf, apply_bitcast, apply_complex
from layers.layer_util import standardize, max_norm, minmax_scale, quantile_norm
from layers.layer_util import log_norm, robust_scale, median_norm, median
from layers.layer_util import float2D_to_cplx1D, cplx1D_to_float2D