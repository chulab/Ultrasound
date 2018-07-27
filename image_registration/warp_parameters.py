"""Functions to set up warping parameters."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import Optional, Union, List

import tensorflow as tf

tfe = tf.contrib.eager


def make_elastic_warp_variable(
    num_control_points: Union[tf.Tensor, int],
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
    tf.Variable of shape [num_control_points, 2].
  '''

  with tf.variable_scope("elastic_warp"):
    if isinstance(initial_offsets_or_scale, List):
      assert len(initial_offsets_or_scale) == num_control_points

      elastic_warp_points = tfe.Variable(
        initial_offsets_or_scale,
        dtype=tf.float32,
        trainable=trainable,
      )

    else:
      assert isinstance(initial_offsets_or_scale, float)
      elastic_warp_points = tfe.Variable(
        tf.random_uniform([num_control_points, 2], minval=-1,
                          maxval=1) * initial_offsets_or_scale,
        dtype=tf.float32,
        trainable=trainable,
      )

  return elastic_warp_points


def make_rotation_warp_variable(
    initial_rotation: Optional[float]=None,
    trainable: Optional[bool]=True,
)-> tf.Tensor:
  """Sets up rotation warp variable."""
  return tfe.Variable(initial_rotation, trainable=trainable)


def make_translation_warp_variable(
    initial_translation: Optional[List[float]]=None,
    trainable: Optional[bool]=True,
) -> tf.Tensor:
  """Sets up translation warp variable."""
  if initial_translation is None:
    initial_translation = [0., 0.]
  return tfe.Variable(initial_translation, trainable=trainable)