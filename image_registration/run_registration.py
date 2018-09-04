"""Runs registration algorithm."""

<<<<<<< HEAD
=======
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

>>>>>>> de63788... Minimal example for yilei.
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import time
<<<<<<< HEAD
import sys
=======
>>>>>>> de63788... Minimal example for yilei.

import loss_utils
import image_field
import warp
import control_point_utils

tfe = tf.contrib.eager

tf.enable_eager_execution()

<<<<<<< HEAD

sys.path.insert(0, '/Users/noah/Documents/CHU/Ultrasound/lib/image_utils/')
import visualization_utils



dataset_path = "/Users/noah/Documents/CHU/Ultrasound/raw_data/heart_rotation"

num_steps = 200
learning_rate = .7
beta = .8
beta_2 = .99
warp_points = (10, 10)
scale = 4.


experiment_name = "9_4_registration"
save_dir = os.path.join("/Users/noah/Documents/CHU/Ultrasound/experiment_results/", experiment_name)
=======
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
>>>>>>> de63788... Minimal example for yilei.

if not os.path.exists(save_dir):
  os.makedirs(save_dir)

images_file = os.path.join(dataset_path, 'raw_images.npy')
data_temp = np.load(images_file).astype(np.float32)

print(data_temp.shape)

<<<<<<< HEAD

# Load first two images.
# =====================
image_a = data_temp[0]
image_b = data_temp[1]

total_size = image_a.shape

xx, yy = np.meshgrid(np.linspace(0, data_temp.shape[1], warp_points[0]),
                     np.linspace(0, data_temp.shape[2], warp_points[1]))
=======
# Load first two images.
image_a = data_temp[0]
image_b = data_temp[1]

size = image_a.shape

xx, yy = np.meshgrid(np.linspace(0, data_temp.shape[1], 10),
                     np.linspace(0, data_temp.shape[2], 10))
>>>>>>> de63788... Minimal example for yilei.

control_points = np.stack([xx, yy], -1)
control_points = np.reshape(control_points, [-1, 2])

<<<<<<< HEAD
control_points = tf.convert_to_tensor(control_points, dtype = tf.float32)

image_a = image_field.load_image(image_a, control_points, -10., [-20., 125.], 1.)
image_b = image_field.load_image(image_b, control_points, 0., [0., 0.], 1.)

center_point = tf.constant([total_size[0] / 2, total_size[1] / 2], dtype=tf.float32)


# Save initial position and warp.
# ==============================

image = image_a.image[tf.newaxis, :, :, tf.newaxis]

# warped_values is list of tf.Tensor for each image
warp_variables = []
warp_values = []

# Rotation.
warp_variables += image_a.get_list_from_variable_dict("rotation")
warp_values += [control_point_utils.project_rotation_on_control_points(
  image_a.control_points, center_point, image_a.get_list_from_variable_dict("rotation")[0])]

# Translation.
warp_variables += image_a.get_list_from_variable_dict("translation")
warp_values += [image_a.get_list_from_variable_dict("translation")[0][tf.newaxis, :]]


warp_values_ = [var[tf.newaxis, :, :] for var in warp_values]

displacement_vectors = tf.zeros_like(image_a.control_points)
for warp_list in warp_values:
    displacement_vectors = displacement_vectors + warp_list

warped_image, dense_warp = warp.sparse_warp(image, image_a.control_points, warp_values_, 1.)

fig,  ax = plt.subplots(2, 1, figsize=(20,10))
visualization_utils.plot_displacement_vectors(image_a.control_points,
                          displacement_vectors.numpy(),
                          [image_a.image.numpy(), warped_image[0, :, :, 0].numpy()],
                                              ax[0])
ax[1].imshow(visualization_utils.multiple_intensity_to_rgb(image_b.image.numpy(), warped_image[0, :, :, 0].numpy()))
fig.savefig(os.path.join(save_dir, "initial_images"))



# Register
# ========

=======
image_a = image_field.load_image(image_a, control_points, 0., [0., 0.], 1.)
image_b = image_field.load_image(image_b, control_points, 0., [0., 0.], 1.)


center_point = tf.constant([total_size[0] / 2, total_size[1] / 2], dtype=tf.float32)


# Register
>>>>>>> de63788... Minimal example for yilei.

global_step = tfe.Variable(0)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                   beta1=beta,
                                   beta2=beta_2,
                                   )

time_start = time.time()

for step in range(num_steps):
  with tfe.GradientTape(persistent=True) as tape:
<<<<<<< HEAD
=======
    scale = 5.
>>>>>>> de63788... Minimal example for yilei.

    # random selection of query points
    # TODO(noah) Select points from overlap of images in eif

    num_query_points = 1000
    query_points = tf.concat(
      [tf.random_uniform([1, num_query_points, 1], 0, 600),
       tf.random_uniform([1, num_query_points, 1], 0, 600)], -1)

    # warped_values is list of tf.Tensor for each image
    warp_variables = []
<<<<<<< HEAD
    warp_values = []

    # Rotation.
    #     warp_variables += image_a.get_list_from_variable_dict("rotation")
    warp_values += [control_point_utils.project_rotation_on_control_points(
      image_a.control_points, center_point,
      image_a.get_list_from_variable_dict("rotation")[0]
    )]

    # Non Rigid
    warp_variables += image_a.get_list_from_variable_dict("non_rigid")
    warp_values += image_a.get_list_from_variable_dict("non_rigid")

    # Translation.
    #     warp_variables += image_a.get_list_from_variable_dict("translation")
    warp_values += [
      image_a.get_list_from_variable_dict("translation")[0][tf.newaxis, :]]

    image = image_a.image[tf.newaxis, :, :, tf.newaxis]
=======
    warp_values =[]

    # Rotation.
    warp_variables += image_a.get_list_from_variable_dict("rotation")
    warp_values += control_point_utils.project_rotation_on_control_points(
      control_points, center_point
    )

    warp_variables += image_a.get_list_from_variable_dict("non_rigid")

    image = image_a.image[tf.newaxis, :, :, tf.newaxis]
    control_points = image_a.control_points[tf.newaxis, :, :]
>>>>>>> de63788... Minimal example for yilei.
    warp_values = [var[tf.newaxis, :, :] for var in warp_values]

    warped_values = warp.get_warp_values(
      image,
      query_points,
<<<<<<< HEAD
      image_a.control_points[tf.newaxis, :, :],
=======
      control_points,
>>>>>>> de63788... Minimal example for yilei.
      warp_values,
      scale,
    )

    image = image_b.image[tf.newaxis, :, :, tf.newaxis]
<<<<<<< HEAD
=======
    control_points = image_b.control_points[tf.newaxis, :, :]
>>>>>>> de63788... Minimal example for yilei.
    warp_values = []

    truth_vals = warp.get_warp_values(
      image,
      query_points,
<<<<<<< HEAD
      image_b.control_points[tf.newaxis, :, :],
=======
      control_points,
>>>>>>> de63788... Minimal example for yilei.
      warp_values,
      scale,
    )

    mse_loss = loss_utils.masked_mse([warped_values, truth_vals])

<<<<<<< HEAD
    warp_loss = loss_utils.warp_loss(tf.reshape(image_a.get_list_from_variable_dict("non_rigid"), [1, warp_points[0], warp_points[1], 2]), 1)

    total_loss =  mse_loss + warp_loss
=======
    warp_loss = loss_utils.warp_loss(tf.reshape(image_a.get_list_from_variable_dict("non_rigid"), [1, 10, 10, 2]), 2)

    total_loss = warp_loss + mse_loss
>>>>>>> de63788... Minimal example for yilei.

  grads = tape.gradient(total_loss, warp_variables)

  optimizer.apply_gradients(
    zip(grads, warp_variables),
    global_step=global_step)

  del tape

time_end = time.time()

print("runtime {}".format(time_end - time_start))

<<<<<<< HEAD

# Save results of registration.
# =============================

image = image_a.image[tf.newaxis, :, :, tf.newaxis]
control_points = image_a.control_points[tf.newaxis, :, :]


# warped_values is list of tf.Tensor for each image
warp_variables = []
warp_values =[]

# Rotation.
warp_values += [control_point_utils.project_rotation_on_control_points(
  image_a.control_points, center_point, image_a.get_list_from_variable_dict("rotation")[0])]

# Translation.
warp_values += [image_a.get_list_from_variable_dict("translation")[0][tf.newaxis, :]]

# Non-rigid
warp_values += image_a.get_list_from_variable_dict("non_rigid")


print(warp_values)
warp_points = tf.zeros_like(warp_values[0])
for warp_list in warp_values:
  warp_points = warp_points + warp_list

warp_values = [var[tf.newaxis, :, :] for var in warp_values]

warped_image, dense_warp = warp.sparse_warp(image, control_points, warp_values, 1.)
initial_image, _ = warp.sparse_warp(image, control_points, warp_values[:-1], 1.)



fig, ax = plt.subplots(2,2, figsize=(15,15))
visualization_utils.plot_displacement_vectors(image_a.control_points,
                                              warp_points,
                          [image_a.image.numpy(), warped_image[0, :, :, 0].numpy()], ax[0, 0])
ax[0, 0].set_title("Optimized warp.")


ax[0, 1].imshow(visualization_utils.multiple_intensity_to_rgb(initial_image[0, :, :, 0].numpy(),
                                                              warped_image[0, :, :, 0].numpy()))
ax[0,1].set_title("Initial and warped image.")

ax[1, 0].imshow(visualization_utils.multiple_intensity_to_rgb(image_b.image.numpy(), initial_image[0, :, :, 0].numpy()))
ax[1, 0].set_title("Initial registration.")

ax[1, 1].imshow(visualization_utils.multiple_intensity_to_rgb(image_b.image.numpy(), warped_image[0, :, :, 0].numpy()))
ax[1, 1].set_title("Optimized registration.")

fig.savefig(os.path.join(save_dir, "warped_image"))
=======
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

>>>>>>> de63788... Minimal example for yilei.

