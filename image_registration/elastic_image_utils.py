"""Utility function for handling `ElasticImage`."""

import tensorflow as tf

from typing import List

from image_registration import elastic_image
from image_registration import warp_utils

_ZERO_POINT = [0., 0]


def reduce_rotation(
    list_of_elastic_images: List[elastic_image.ElasticImage]
):
  """Reorients images to make mean rotation (about center point) 0.

  This function calculates the mean rotation of all images in the provided list
  and  updates the translation and rotation of each image to make the mean
  rotation 0.

  Args:
    list_of_elastic_images: List over which to calculate and reduce the mean
    rotation.

  Returns:
    `tf.Operation` to perform update of image rotations and translations.
  """

  # Calculate mean rotation.
  rotations = [image.rotation for image in list_of_elastic_images]
  mean_rotation = tf.reduce_mean(rotations)

  # Remove mean rotation from each image.
  rotation_update = [
    tf.assign_add(image.rotation, - mean_rotation)
    for image in list_of_elastic_images
  ]

  # Calculate the change in the translation vector due to a rotation about the center point.
  translations = [
    warp_utils.rotate_points(
      image.translation[tf.newaxis, :],
      tf.convert_to_tensor(_ZERO_POINT),
      -mean_rotation)[0, :]
    for image in list_of_elastic_images
  ]

  # Update the translation vector.
  translation_update = [
    tf.assign_add(image.translation, translation)
    for image, translation
    in zip(list_of_elastic_images, translations)
  ]

  return rotation_update + translation_update


def reduce_translation(
    list_of_elastic_images: List[elastic_image.ElasticImage]
):
  """Calculate and subtracts mean translation from each image in List.

  Args:
    list_of_elastic_images: See documentation for `reduce_rotation`.

  Returns:
    `tf.Operation` to perform update of image rotations and translations.

  """
  translations = [image.translation for image in list_of_elastic_images]
  mean_translation = tf.reduce_mean(translations, axis=0)

  return [tf.assign_add(image.translation, -mean_translation)
          for image in list_of_elastic_images
          ]
