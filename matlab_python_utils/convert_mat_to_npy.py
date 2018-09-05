import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt

ang = np.arange(1,10)


def convert_mat_to_numpy(data_location:str,
                         file_name: str,
                         mat_name: str = None,
                         save_location:str = None,
):
  """Converts an array stored in a `.mat` file to a numpy array and saves.

  Args:
    data_location: Path to mat file.
    file_name: name of mat file.
    mat_name: name of array in mat file.
    save_location:
  """
  file_name = os.path.join(data_location, file_name)

  temp_data = sio.loadmat(file_name)

  raw_data = temp_data['movie_int']

  raw_data = np.swapaxes(raw_data,2,0)
  raw_data = np.swapaxes(raw_data,1,2)

  print(raw_data.shape)

  sample_image_file = os.path.join(data_location, "example")

  for i in range(raw_data.shape[0]):
      plt.imsave(sample_image_file+str(i), raw_data[i])

  print(raw_data[0])

  # ### need to rescale to 0-255
  # max = np.amax(raw_data)
  # scaled_data = (raw_data*255/max).astype(np.uint8)
  #
  # print(scaled_data.shape)
  #
  #
  # print(scaled_data.shape)
  # # zoom_ratio = (512 / 1456) * (28 / 23)
  # # im_zoomed = ndimage.zoom(scaled_data, zoom=(1, zoom_ratio, 1), order=1)
  save_file = os.path.join(data_location,"raw_images")
np.save(save_file, raw_data)


if __name__ == "__main__":
  convert_mat_to_numpy(
  data_location = "../../raw_data/balloon_wrist/",
  file_name= "movie_water.mat"
  )