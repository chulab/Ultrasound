"""Utilities for processing image arrays.

All utilities use array indexing (also known as i-j indexing as opposed to
x-y indexing). For example, the positions [1,2] and [4, 3] will correspond to:

  . . . . .
  . . x . .
  . . . . .
  . . . . .
  . . . x .
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Union, List, Tuple

import tensorflow as tf

def calculate_pad_size(
        array_shape: Union[tf.Tensor, List[int]],
        new_shape: Union[tf.Tensor, List[int]],
        offsets: Optional[Union[tf.Tensor, List[int]]]=None,
        ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  '''Calculate padding values to pad array to new shape.

  Args:
    array_shape: 1D tf.Tensor or List of form [height, width].
    new_shape: 1D tf.Tensor or List of form [new_height, new_width].
    offsets: 1D tf.Tensor or List of form [height_offset, width_offset].

  Returns:
    padding: tf.Tensor of shape [2, 2].
    corner_position: tf.Tensor of shape [2, 2, 2] containing the new position
      of each corner of the array given the calculated paddings.  The
      dimensions of the tensor index respectively
      [height, width, corner_position].  Explicitly the top left corner
      will be:
        height, width = [0, 0, :]
    center_position: tf.Tensor of form [height, width].
  '''

  array_shape = tf.convert_to_tensor(array_shape)
  new_shape = tf.convert_to_tensor(new_shape)

  padding = tf.cast( (new_shape - array_shape)/ 2, tf.int32)

  if offsets:
    offsets = tf.convert_to_tensor(offsets)
    pad_top_left = padding + offsets
    pad_bottom_right = padding - offsets
  else:
    pad_top_left = pad_bottom_right = padding

  padding = tf.stack([pad_top_left, pad_bottom_right], 1)

  # Calculate corner positions.
  corner_position = pad_top_left[tf.newaxis, tf.newaxis, :] + \
    tf.stack([
    tf.stack([[0, 0], [0, array_shape[1]]], axis = 0),
    tf.stack([[array_shape[0], 0], array_shape], axis=0),
      ])

  # Calculate center_point of padded array.
  center_position = tf.cast(corner_position[0, 0], tf.float32) \
                    + tf.cast(array_shape, tf.float32) /  2

  return padding, corner_position, center_position
