"""Utilities for generating visualizations."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import os
import matplotlib.pyplot as plt


def multiple_intensity_to_rgb(im_1: np.ndarray,
                              im_2: np.ndarray = None,
                              im_3: np.ndarray = None
                              ):
  """Creates an RGB array from 3 intensity maps.

  Given 1 - 3 2D arrays, this function generates an RGB image array with values
  normalized between 0. and 1. The values in the array are normalized to the
  maximum value across all input arrays.

  Args:
    im_1: np.ndarray with shape [H, W].
    im_2: np.ndarray with same shape as `im_1`
    im_3: np.ndarray with same shape as `im_1`

  Returns:
    np.ndarray with shape [H, W, 3].
  """

  im_1 = im_1.astype(np.float32)

  if im_2 is not None:
    im_2 = im_2.astype(np.float32)
  else:
    im_2 = np.zeros_like(im_1)
  if im_3 is not None:
    im_3 = im_3.astype(np.float32)
  else:
    im_3 = np.zeros_like(im_1)

  im_tot = np.stack([im_1, im_2, im_3], axis=-1)

  return im_tot / np.amax(im_tot)


def plot_displacement_vectors(
    control_points: np.ndarray,
    displacement_vector: np.ndarray,
    arrow_scale=1.,
    images=None,
    axes=None, ):
  """Plots a vector field given control points and displacement vectors.

  The displacement vector measures the displacement of a point from the old
  image point to the grid point.

  Args:
    control_points: Array of shape [N, 2]
    displacement_vector: Array of shape [N, 2]
    arrow_scale: Scales the length of the displayed arrows relative to the grid size.
        I.e. `arow_scale=1` means that a displacement vector of .5 will have
        an arrow of length 2. See documentation for `matplotlib.pyplot.quiver` for
        more information.
    image: Optional List of image arrays of shape [H, W] to display behind
    vector field.
    axes: Axes on which to plot.

  Raises:
    ValueError: If `control_points` or `displacement_vectors` have improper
      shape.
  """
  if not control_points.shape == displacement_vector.shape:
    raise ValueError("`control_points` and `displacement_vector` must have"
                     "the same shape, got {}".format(
      [control_points, displacement_vector]))

  X = control_points[:, 0]
  Y = control_points[:, 1]

  U = displacement_vector[:, 0]
  V = displacement_vector[:, 1]

  if axes is not None:
    if images is not None:
      axes.imshow(visualization_utils.multiple_intensity_to_rgb(*images))
    axes.quiver(Y, X, V, U, angles='xy', pivot='tip', color='r',
                scale=arrow_scale, scale_units='x', units='x', width=3)

  else:
    if images is not None:
      plt.imshow(visualization_utils.multiple_intensity_to_rgb(*images))
    plt.quiver(Y, X, V, U, angles='xy', pivot='tip', color='r',
               scale=arrow_scale, scale_units='x', units='x', width=3)
    quiver(Y, X, V, U, angles='xy', pivot='tip', color='r')
