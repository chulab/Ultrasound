"""Runs registration algorithm."""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time

import loss_utils
import image_field
import warp
import control_point_utils

tfe = tf.contrib.eager

tf.enable_eager_execution()

dataset_path = "/Users/noah/Documents/CHU/Ultrasound/raw_data/heart_rotation"

num_steps = 200
learning_rate = .15
beta = .4
beta_2 = .99
warp_points = (8, 8)
initial_scale = 7.
scale_tuner_alpha = 1.2
elastic_weight = 1.
elastic_weight_scale = 0.25057652344002182

save_dir = "/Users/noah/Documents/CHU/Ultrasound/experiment_results/8_22_registration"

if not os.path.exists(save_dir):
  os.makedirs(save_dir)

images_file = os.path.join(dataset_path, 'raw_images.npy')
data_temp = np.load(images_file).astype(np.float32)

print(data_temp.shape)

# Load first two images.
image_a = data_temp[0]
image_b = data_temp[1]

size = image_a.shape

xx, yy = np.meshgrid(np.linspace(0, data_temp.shape[1], 10),
                     np.linspace(0, data_temp.shape[2], 10))

control_points = np.stack([xx, yy], -1)
control_points = np.reshape(control_points, [-1, 2])

image_a = image_field.load_image(image_a, control_points, 0., [0., 0.], 1.)
image_b = image_field.load_image(image_b, control_points, 0., [0., 0.], 1.)


center_point = tf.constant([total_size[0] / 2, total_size[1] / 2], dtype=tf.float32)


# Register

global_step = tfe.Variable(0)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=beta,
                                   beta2=beta_2,
                                   )

time_start = time.time()

for step in range(num_steps):
  with tfe.GradientTape(persistent=True) as tape:
    scale = 5.

    # random selection of query points
    # TODO(noah) Select points from overlap of images in eif

    num_query_points = 1000
    query_points = tf.concat(
      [tf.random_uniform([1, num_query_points, 1], 0, 600),
       tf.random_uniform([1, num_query_points, 1], 0, 600)], -1)

    # warped_values is list of tf.Tensor for each image
    warp_variables = []
    warp_values =[]

    # Rotation.
    warp_variables += image_a.get_list_from_variable_dict("rotation")
    warp_values += control_point_utils.project_rotation_on_control_points(
      control_points, center_point
    )

    warp_variables += image_a.get_list_from_variable_dict("non_rigid")

    image = image_a.image[tf.newaxis, :, :, tf.newaxis]
    control_points = image_a.control_points[tf.newaxis, :, :]
    warp_values = [var[tf.newaxis, :, :] for var in warp_values]

    warped_values = warp.get_warp_values(
      image,
      query_points,
      control_points,
      warp_values,
      scale,
    )

    image = image_b.image[tf.newaxis, :, :, tf.newaxis]
    control_points = image_b.control_points[tf.newaxis, :, :]
    warp_values = []

    truth_vals = warp.get_warp_values(
      image,
      query_points,
      control_points,
      warp_values,
      scale,
    )

    mse_loss = loss_utils.masked_mse([warped_values, truth_vals])

    warp_loss = loss_utils.warp_loss(tf.reshape(image_a.get_list_from_variable_dict("non_rigid"), [1, 10, 10, 2]), 2)

    total_loss = warp_loss + mse_loss

  grads = tape.gradient(total_loss, warp_variables)

  optimizer.apply_gradients(
    zip(grads, warp_variables),
    global_step=global_step)

  del tape

time_end = time.time()

print("runtime {}".format(time_end - time_start))

image = image_a.image[tf.newaxis, :, :, tf.newaxis]
control_points = image_a.control_points[tf.newaxis, :, :]
warp_values = [var[tf.newaxis, :, :] for var in warp_variables]

warped_image, dense_warp = warp.sparse_warp(image, control_points, warp_values, 1.)

fig, ax = plt.subplots(2,2)
ax[0, 0].imshow(image_a.image)
ax[0, 1].imshow(warped_image[0, :, :, 0])
ax[1, 0].imshow(image_b.image - image_a.image)
ax[1, 1].imshow(image_b.image - warped_image[0, :, :, 0])
fig.savefig(os.path.join(save_dir, "warped_image"))

print(warp_values)


