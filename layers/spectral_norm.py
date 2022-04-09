import tensorflow as tf
from _utils.tensor import shape_list

def apply_spectral_norm(x):
  """Normalizes x using the spectral norm.
  The implementation follows Algorithm 1 of
  https://arxiv.org/abs/1802.05957. If x is not a 2-D Tensor, then it is
  reshaped such that the number of channels (last-dimension) is the same.
  Args:
    x: Tensor with the last dimension equal to the number of filters.
  Returns:
    x: Tensor with the same shape as x normalized by the spectral norm.
    assign_op: Op to be run after every step to update the vector "u".
  """
  weights_shape = shape_list(x)
  other, num_filters = tf.reduce_prod(weights_shape[:-1]), weights_shape[-1]

  # Reshape into a 2-D matrix with outer size num_filters.
  weights_2d = tf.reshape(x, (other, num_filters))

  # v = Wu / ||W u||
  u = tf.Variable(tf.compat.v1.truncated_normal(shape=[num_filters, 1]), 
                  name="u", trainable = False)
  v = tf.nn.l2_normalize(tf.matmul(weights_2d, u))

  # u_new = vW / ||v W||
  u_new = tf.nn.l2_normalize(tf.matmul(tf.transpose(v), weights_2d))

  # s = v*W*u
  spectral_norm = tf.squeeze(
      tf.matmul(tf.transpose(v), tf.matmul(weights_2d, tf.transpose(u_new))))

  # set u equal to u_new in the next iteration.
  assign_op = tf.compat.v1.assign(u, tf.transpose(u_new))
  return tf.divide(x, spectral_norm), assign_op