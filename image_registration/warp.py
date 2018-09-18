"""Defines warping functions."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import List

import tensorflow as tf

from modified_tf.modified_sparse_image_warp import sparse_image_warp
from modified_tf.modified_dense_image_warp import dense_image_warp, \
  _interpolate_bilinear
from modified_tf.modified_interpolate_spline import interpolate_spline

tfe = tf.contrib.eager


def sparse_warp(image: tf.Tensor,
                control_points: tf.Tensor,
                warp_values: List[tf.Tensor],
                scale: int = 1, ):
  """Warps image using a sparse set of displacements.

  Optionally rescales image before performing warp. This resizing is
  performed by averaging the pixel intensities within each square region of
  size (2 ^ `scale`).

  Args:
      image: tf.Tensor of shape [B, H, W, C]
      control_points: tf.Tensor of shape [B, num_control_points, 2].
      warp_values: tf.Tensor of shape [B, num_control_points, 2].
      scale: int tf.Tensor. Factors of 2.

  Returns:
      warped image: image.  Tensor of shape [N, H, W, C]
      dense warp: dense warp array. Tenosr of shape [N, H, W, 2]
  """

  with tf.variable_scope("sparse_warp"):

    warp_points = total_warp_points(warp_values, control_points.shape)
    destination_control_points = warp_points + control_points

    if scale != 1.:
      control_ponts = rescale_points(control_points, scale)
      destination_control_points = rescale_points(
        destination_control_points, scale)

      new_image = rescale_image_using_pooling(image, scale)

    else:
      new_image = image

  warped, dense_mesh = sparse_image_warp(new_image,
                                         control_points,
                                         destination_control_points,
                                         interpolation_order=2,
                                         )

  return warped, dense_mesh


def dense_warp(image,
               warp_values):
  """Warps image.

  Args:
      image: tf.Tensor of shape [B, H, W, C]. Float dtype.
      warp_values: list of tf.Tensor.  Each Tensor muse have shape [H,W,2]
  Returns:
      warped_images:
  """

  with tf.variable_scope("dense_warp"):
    image_sz = image.shape
    control_point_shape = tf.TensorShape([image_sz[0], image_sz[1], image_sz[2], 2])
    warp_points = total_warp_points(warp_values, control_point_shape)

    image = tf.cast(image, tf.float32)
    warp_points = tf.cast(warp_points, image.dtype)
    return dense_image_warp(image, warp_points)


def warp_query(image: tf.Tensor,
               query_points: tf.Tensor,
               control_points: tf.Tensor,
               warp_values: List[tf.Tensor],
               scale: int = 1, ):
  """Gets value of warped image at query points in transformed coordinates.

  This function performs a warp and query in the following steps:
    1. Resize image according to `scale`.
    2. Rescale query, control, and warp values according to `scale`.
    3. Warp images according to `control_points` and `warp_values`.
    4. Interpolate value of warped image at `query_points`.

  Note `query_points` should be provided in the coordinates of the
  warped image.

  `scale` affects the size of the image (and point coordinates) as
  `image.shape = image.shape / scale.`.  Explicitly:

    # image.shape = [1, 10, 10, 2]
    # scale = 2
    # scaled_image = [1, 5, 5, 2]

  The warp is performed by displacing `control_points` according to the
  `warp_values`.

  Explicitly:
    image = ...
    control_points = [[[3, 4], ... ]]
    warp_values = [[[[1, 1]]] , ... ]
    query = warp_query(image, [[[4, 5]]], control_points, warp_values, ... )
    # query = image[:, 3, 4, :]

  For further information see documentation for `modified_sparse_image_warp`.

  Args:
      image: `tf.Tensor` of shape
        `[batch_size, x_dimension, y_dimension, channel_count]`.
      query_points: `tf.Tensor` compatible with shape
        `[batch_size, query_count, 2]`. Corresponds to coordinates in the
        space of the warped image that will be queried.
      control_points: `tf.Tensor` of shape
        `[batch_size, control_point_count, 2]`. Control points in the
        coordinates of the moving image.
      warp_values: List of `tf.Tensor`. Each `tf.Tensor` should have shape
        compatible with `control_points.shape`.  Each `tf.Tensor` is a
        vector from the control points in unwarped moving image to the
        new position in the warped moving image.
      scale: float. Controls amount that coordinates are rescaled before
        querying.  A higher `scale` corresponds to reducing the dimension
        of the sampled image.

  Returns:
    `tf.Tensor` of shape `[batch_size, query_count, channel_count]`. The
    intensity of the warped image at the queried locations.
  """

  with tf.variable_scope("warp_query"):
    warp_points = total_warp_points(warp_values, control_points.shape)
    destination_control_points = warp_points + control_points

    if scale != 1:
      # Rescale image, control points, and query points to correspond to
      # rescaled coordinate system.
      control_points = rescale_points(control_points, scale)
      destination_control_points = rescale_points(destination_control_points,
                                                  scale)
      query_points = rescale_points(query_points, scale)
      image = rescale_image_using_pooling(image, scale)

  return _warp_query(image,
                     query_points,
                     control_points,
                     destination_control_points,
                     )

def _warp_query(
    image,
    query_points,
    control_point_locations,
    destination_control_points,
    interpolation_order=1):
  """Gets value of warped image at queried points.

  Args:
    image: See documentation for `warp_query`.
    query_points: See documentation for `warp_query`.
    control_point_locations: `tf.Tensor` of shape
      `[batch_size, point_count, 2]`.  Describes points that paramaterize
      image warp.
    destination_control_points: `tf.Tensor` of same shape as
      `control_point_locations`.
    interpolation_order: See documentation for `sparse_image_warp`.

  Returns:
    Value of warped image at query_points.  `tf.Tensor` of shape
      `[batch_size, query_count, channel_count]`.
  """
  # Cast image, warp, and query points to correct dtype.
  image = tf.cast(image, tf.float32)
  query_points = tf.cast(query_points, tf.float32)
  control_point_locations = tf.cast(control_point_locations, tf.float32)
  destination_control_points = tf.cast(destination_control_points, tf.float32)

  # Compute `flow`.  The flow is the set of vectors from
  # `control_point_locations` to `destination_control_points`.
  control_point_flows = (
      destination_control_points - control_point_locations)

  # Interpolate flow at the query point locations.
  flows = interpolate_spline(
    destination_control_points, control_point_flows,
    query_points, interpolation_order)
  query_point_flows = query_points - flows

  # Interpolate value of the image at the queried location by sampling into
  # the original image.
  return _interpolate_bilinear(image, query_point_flows)


def rescale_points(points: tf.Tensor,
                   scale: float,
                   ):
  """Scales a set of index points by 1 / `scale`.

  This function changes the scale of points which are used to paramaterize
  warp points in an image or a displacement mesh.

  Args:
    points: `tf.Tensor` of shape `[num_points, 2]`.
    scale: Float by which to rescale `points`.

  Returns:
    `tf.Tensor` of same shape as `points`.
  """
  scale = tf.convert_to_tensor(scale)
  scale = tf.cast(scale, points.dtype)
  return points / scale


def rescale_image_using_pooling(image: tf.Tensor,
                                scale: int, ):
  """Rescales the size of an image using an average pooling operation.

  Optionally rescales image before performing warp. This resizing is
  performed by averaging the pixel intensities within each square region of
  size (2 ^ `scale`).

  Args:
    image: `tf.Tensor` of shape `[B, H, W, C]`.
    scale: Amount by which to rescale `image`.

  Returns:
   `tf.Tensor` of shape `[B, H / scale , W, C]`.
  """
  image = tf.cast(image, tf.float32)
  return tf.nn.pool(image, [scale, scale], "AVG", "VALID",
                    strides=[scale, scale])


def total_warp_points(warp_list: List[tf.Tensor],
                      control_point_shape: tf.TensorShape):
  """Generates a single warp point tensor from a list of warp tensors.

  A single waro tensor is a `tf.Tensor` containing an X and Y displacement for
  each warp point in an image.  This function combines several warp tensors
  linearly to create a cumulative warp tensor.

  Explicitly:
    warp_list = [warp_a, warp_b]
    warp_points = total_warp_points(warp_list)
    # warp_points[i, j, :] = warp_a[i, j, k] + warp_b[i, j, k]

  Args:
    warp_list: List of tf.Tensor. Each `tf.Tensor` has shape
    compatible with `control_point_shape`.
    control_point_shape: `tf.TensorShape` describing control points on image.

  Returns:
    `tf.Tensor` of same shape `[B, point_count, 2]`.

  Raises:
    ValueError: If a `tf.Tensor` in `warp_list` does not have shape
    compatible with `control_point_shape`.
  """
  if not all(tensor.shape.is_compatible_with(control_point_shape) for tensor
             in warp_list):
    raise ValueError("All warp tensors should have shape compatible with {}, "
                     "got {}".format(control_point_shape,
                                     [tensor.shape for tensor in warp_list]
                                     ))

  total_warp_tensor = tf.zeros(control_point_shape, dtype=warp_list[0].dtype)

  for warp_tensor in warp_list:
    total_warp_tensor = total_warp_tensor + warp_tensor

  return total_warp_tensor
