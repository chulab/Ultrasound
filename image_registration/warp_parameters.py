"""Functions to set up warping parameters."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import Optional, Union, List

import tensorflow as tf

tfe = tf.contrib.eager


def make_elastic_warp_variable(
    control_points_shape: tf.TensorShape,
    initial_offsets_or_scale: Union[float, List[List[float]]],
    trainable: Optional[bool] = True,
) -> tf.Tensor:
  '''Creates variable to paramaterize non-rigid warp.

  Args:
    num_control_points: tf.Tensor or int.
    scale:  Determines scale of random variation from `initial_guess`.
    initial_offsets:  Array shape [N, 2] where N is num_control_points. Can
      choose either `scale` or `initial_offsets` to initialize warp.
    trainable: Set variables to be trainable.

  Returns:
    tf.Variable of shape [batch_size, control_point_count, 2].
  '''
  with tf.variable_scope("elastic_warp"):
    if isinstance(initial_offsets_or_scale, List):
      elastic_warp_points = tfe.Variable(
        initial_offsets_or_scale,
        dtype=tf.float32,
        trainable=trainable,
      )

    else:
      assert isinstance(initial_offsets_or_scale, float)
      elastic_warp_points = tfe.Variable(
        tf.random_uniform(control_points_shape, minval=-1,
                          maxval=1) * initial_offsets_or_scale,
        dtype=tf.float32,
        trainable=trainable,
      )

  return elastic_warp_points


def make_rotation_warp_variable(
  initial_rotation: float,
  trainable: Optional[bool]=True,
)-> tf.Tensor:
  """Sets up rotation warp variable.

  The rotation warp variable parametrizes the rotation of a moving image
  in degrees.

  Args:
    initial_rotation: Float, parametrizes initial rotation in degrees.

  Returns:
    tf.Variable of shape `[batch_size]`

  Raises:
    ValueError: If `initial_rotation` has a shape not compatible with
      `[batch_size]`.
  """
  if not isinstance(initial_rotation, (float, int)) :
    raise ValueError("Intial rotation must be a scalar (float or int) "
                     ", got {}".format(initial_rotation))

  return tfe.Variable(initial_rotation, trainable=trainable)


def make_translation_warp_variable(
    initial_translation: Optional[List[float]]=None,
    trainable: Optional[bool]=True,
) -> tf.Tensor:
  """Sets up translation warp variable.

  Returns:
    `tf.Tensor` of shape `[batch_size, 2]`.
  """
  if initial_translation is None:
    initial_translation = [0., 0.]
  if not (len(initial_translation) == 2 and
      isinstance(initial_translation[0], (float, int))) :
    raise ValueError("Initial translation must be a List of form "
                     "`[initial_x, initial_y]`, got "
                     "{}".format(initial_translation))
  return tfe.Variable(initial_translation, trainable=trainable)