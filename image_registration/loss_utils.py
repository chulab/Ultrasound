"""Defines loss functions for performing image registration."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
from typing import List

tfe = tf.contrib.eager


def masked_mse(
    comparison_list: List[tf.Tensor],
    mask_list: List[tf.Tensor] = None,
) -> tf.Tensor:
  """Calculates the MSE between all unmasked elements.

  masked_mse compares elements across a set of tensors of the same shape.

  In general let T_1, ... T_n be tensors to compared.  Let each tensor to
  have shape [D_1, ..., D_i].  Each pair of elements with the same index
  across tensors are compared and included in the MSE error unless mask
  tensor is 0 on either element.

  Let {elements} be the set of possible indices into any tf.Tensor in
  comparison_list or mask_list.

  Let T_i_e be the value of value tensor T_i at position e \in {elements}.
  Let M_i_e be the value of mask tensor M_i at position e \in {elements}.

  Explicitly, masked_mse calculates:

  \sum_{i,i*}{\sum_{e}{T_i_e - T_j_e}} where i,i* \in n, i=!i* and
   e \in {elements} and M_i_e[1]==M_i_e[1]=1.

  Args:
      comparison_list: List of tf.Tensor objects with shape (D_1, ... D_i).
          Values in identical indices will be compared
      masks: Optional List of binary tf.Tensor objects with the same shape
          as Tensors in comparison_list. Each mask Tensor defines the elements
          to be used for mse calculation.  If the mask at a location is 1, then
          the corresponding value is included in loss calculation.

  Returns:
      Calculate mean squared error.
  """

  values = tf.stack(comparison_list)

  values = tf.reshape(values, [values.shape[0], -1])

  differences = tf.expand_dims(values, 0) - tf.expand_dims(values, 1)

  # Remove repeated elements from broadcasting combinations.
  differences = tf.transpose(
    tf.matrix_band_part(tf.transpose(differences, (2, 0, 1), ), -1, 0),
    (1, 2, 0), )
  
  # Mask invalid MSE entries is a mask is provided.
  if mask_list:
      masks = tf.stack(mask_list)
      masks = tf.reshape(masks, [masks.shape[0], -1])
      overlaps = tf.expand_dims(masks, 0) * tf.expand_dims(masks, 1)
      differences = differences * overlaps
      return tf.reduce_sum(tf.pow(differences, 2)) / \
             tf.count_nonzero(differences, dtype=tf.int32)
  else:
      return tf.reduce_sum(tf.pow(differences, 2)) / tf.cast(tf.size(values), tf.float32)


def warp_loss(dense_warp, correlation_range):
  """Constructs regularization loss on dense warp matrix.

  Builds loss based on 2D `smoothness` of warp parameters.

  Args:
      dense_warp: Tensor of shape [N, W, H, 2].  Where N is batch size.
      correlation_range: Number of pixels to include in convolution

  Returns:
      warp_loss:
  """

  array_size = correlation_range * 2 + 1

  ham2d = tf.sqrt(tf.einsum('i,j->ij',
                            tf.contrib.signal.hamming_window(array_size),
                            tf.contrib.signal.hamming_window(array_size)))

  center_replacement = tf.pad(tf.ones([1, 1]),
                              [[correlation_range, correlation_range],
                               [correlation_range, correlation_range]])
  correlation_kernel = ham2d - center_replacement * tf.reduce_sum(ham2d)
  correlation_kernel = correlation_kernel / correlation_kernel[
    correlation_range, correlation_range]
  correlation_kernel = tf.stack(
    [correlation_kernel, tf.zeros_like(correlation_kernel)], axis=2)
  correlation_kernel = tf.stack(
    [correlation_kernel, correlation_kernel[:, :, ::-1]], axis=3)

  def _grid_convolve(tensor, correlation_kernel):
    """Convolves tensor with kernel.

    Args:
        tensor: Tensor of shape [N, W, H, 2].
        correlation_kernel: Kernel to evaluate correlation.  Shape [C, C, 2, 2]

    Returns:
        convolved: tensor convolved with correlation_kernel
    """
    # Symmetric padding ensures that tensors along the edge are not penalized
    pad_size = correlation_range

    tensor = tf.pad(tensor,
                    [[0, 0], [pad_size, pad_size], [pad_size, pad_size],
                     [0, 0]], mode="SYMMETRIC")
    return tf.nn.conv2d(tensor, correlation_kernel, strides=[1, 1, 1, 1],
                        padding="VALID")

  # Compute loss for warp given kernel.
  convolved = _grid_convolve(dense_warp, correlation_kernel)
  convolved = tf.pow(convolved, 2)
  loss = tf.reduce_mean(convolved)
  return loss
