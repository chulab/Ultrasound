"""Defines warping functions."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from typing import List

import tensorflow as tf

from modified_sparse_image_warp import sparse_image_warp
from modified_dense_image_warp import dense_image_warp, _interpolate_bilinear
from modified_interpolate_spline import interpolate_spline

tfe = tf.contrib.eager


def sparse_warp(image: tf.Tensor,
                control_points: tf.Tensor,
                warp_values: List[tf.Tensor],
                scale: float = 1., ):
    """Warps image using a sparse set of displacements.

    Args:
        image: tf.Tensor of shape [B, H, W, C]
        control_points: tf.Tensor of shape [B, num_control_points, 2].
        warp_values: tf.Tensor of shape [B, num_control_points, 2].
        scale: int tf.Tensor. Factors of 2
    Returns:
        warped image: image.  Tensor of shape [N, H, W, C]
        dense warp: dense warp array. Tenosr of shape [N, H, W, 2]
    """

    with tf.variable_scope("warp"):

        scale = tf.convert_to_tensor(scale)
        scale = tf.cast(scale, tf.float32)
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        with tf.variable_scope("warp_points"):
            warp_points = tf.zeros_like(control_points)

            for warp_list in warp_values:
                warp_points = warp_points + warp_list

            destination_control_points = warp_points + control_points

            source_points = control_points

        if scale != 1.:
            source_points = control_points / (2 ** scale)
            destination_control_points = destination_control_points / (2 ** scale)

            # new_size = tf.floordiv(tf.shape(image)[1:3], tf.cast(scale, tf.int32))
            # new_image = tf.image.resize_images(image, new_size, method=tf.image.ResizeMethod.BILINEAR)

            new_image = tf.nn.pool(image, [scale, scale], "AVG", "VALID",
                       strides=[scale, scale])
        else:
            new_image = image

    warped, dense_mesh = sparse_image_warp(new_image,
                             source_points,
                             destination_control_points,
                             interpolation_order=2,
                             )

    return  warped, dense_mesh


def dense_warp(image,
               warp_values):
    """Warps image.

    Args:
        image: tf.Tensor of shape [B, H, W, C]. Float dtype.
        warp_values: list of tf.Tensor.  Each Tensor muse have shape [H,W,2]
    Returns:
        warped_images:
    """


    with tf.variable_scope("warp"):
        image = tf.convert_to_tensor(image)

        image = tf.cast(image, tf.float32)

        image_sz = tf.shape(image)

        print(image_sz)

        warp_points = tf.zeros([image_sz[1], image_sz[2], 2])

        for warp_matrix in warp_values:
            warp_points = warp_points + warp_matrix

    return dense_image_warp(image, warp_points)


def get_warp_values(image: tf.Tensor,
                    query_points:tf.Tensor,
                    control_points: tf.Tensor,
                    warp_values: List[tf.Tensor],
                    scale: float = 1., ):
    """Performs sparse warp on image.

    Args:
        image:
        control_points:
    Returns:
    """

    with tf.variable_scope("warp"):

        scale = tf.convert_to_tensor(scale)
        scale = tf.cast(scale, tf.float32)
        image = tf.convert_to_tensor(image, dtype=tf.float32)

        with tf.variable_scope("warp_points"):
            warp_points = tf.zeros_like(control_points)

            for warp_list in warp_values:
                warp_points = warp_points + warp_list

            destination_control_points = warp_points + control_points

            source_points = control_points

        if scale != 1.:
            source_points = control_points / scale
            destination_control_points = destination_control_points / scale

            new_size = tf.floordiv(tf.shape(image)[1:3], tf.cast(scale, tf.int32))
            new_image = tf.image.resize_images(image, new_size, method=tf.image.ResizeMethod.BILINEAR)

            query_points = query_points / scale

        else:
            new_image = image

    return _get_warp_value(new_image,
                           query_points,
                           source_points,
                           destination_control_points
                           )

def _get_warp_value(image,
                   query_points,
                   source_control_point_locations,
                   dest_control_point_locations,
                   interpolation_order=1,
                   regularization_weight=0.0, ):
    """Gets value of warped image at queried points.

    Args:
        image: 4D tensor of shape [B, H, W, C]
        query_points: Tensor of shape [batch, N, 2]
        source_control_point_locations:
        dest_control_point_locations:
        interpolation_order:
        regularization_weight:

    Returns:
        Value of warped image at query_points.  tf.Tensor of shape [b, n, 1]

     """

    image = tf.convert_to_tensor(image)
    source_control_point_locations = tf.convert_to_tensor(
        source_control_point_locations)
    dest_control_point_locations = tf.convert_to_tensor(
        dest_control_point_locations)
    query_points = tf.convert_to_tensor(query_points)

    control_point_flows = (
        dest_control_point_locations - source_control_point_locations)
    flows = interpolate_spline(
        dest_control_point_locations, control_point_flows,
        query_points, interpolation_order, regularization_weight)
    query_point_flows = query_points - flows
    # calculate warp values
    return _interpolate_bilinear(image, query_point_flows)


