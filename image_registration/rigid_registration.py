"""Functions to prerform reigid registration of images.

Rigid registration is a pre-processing step for the non-rigid registration
performed by the `non-rigid-registration` functions."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from skimage import feature
from skimage import exposure
from skimage import transform

from typing import List


def relative_rotation_and_translation(
  images: List[np.ndarray],
  theta_range: List[float],
  angle_count: int,
):
  """Finds relative rotation and translation between adjacent images in list.

  Wraps `rotation_and_translation` to compute the rigid translation.

  For orientation see `rotation_and_translation`.

  Args:
    images: List of np.ndarray images.  Must be of same shape.
    theta_range: See documentation for `rotation_and_translation`.
    angle_count: See documentation for `angle_count`.

  Returns:
    translations: List of translations.
    rotations: List of rotations.

  Raises:
    ValueError: If not all images are the same shape.
  """
  if not all(image.shape == images[0].shape for image in images):
    raise ValueError("All images must be same shape, got {}".format([image.shape for image in images]))

  pairs_of_images = zip(images, images[1:])

  translations, rotations = zip(
    *[rotation_and_translation(image_a, image_b, theta_range, angle_count) for
      image_a, image_b in pairs_of_images]
  )

  return translations, rotations

def rotation_and_translation(
    image_a: np.ndarray,
    image_b: np.ndarray,
    theta_range: List[float],
    angle_count: int=25,
):
  """Finds displacement and rotation of image_b relative to image_a.

  Returns an estimated rotation and translation of image b from a.
  The rotation is measured clockwise from a.

  Args:

  Returns:

  """

  # Normalize intensities of images.
  image_a = exposure.rescale_intensity(image_a)
  image_b = exposure.rescale_intensity(image_b)

  running_angle = []
  running_translation = []
  running_error = []

  # Search over rotations.
  for rotation_degree in np.linspace(*theta_range, angle_count):
    translation, error_temp, _ = feature.register_translation(
      image_a,
      transform.rotate(image_b, rotation_degree),
    )
    running_translation += [translation]
    running_angle += [rotation_degree]
    running_error += [error_temp]

  min_element = np.argmin(running_error)

  return running_angle[min_element], running_translation[min_element]