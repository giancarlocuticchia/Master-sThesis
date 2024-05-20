import cv2
from datetime import datetime
import glob
from PIL import Image
import math
import numpy as np
import os
import pandas as pd
import pytz
import shutil
from skimage.metrics import structural_similarity as skimage_ssim
import subprocess
import time

import stardist
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from stardist.matching import matching
from stardist.plot import render_label


## Retrieve a sorted list of the paths of all the files with a certain extension inside a directory
def retrieve_filepaths(directory, file_extension=""):
  return sorted(glob.glob(os.path.join(directory, '*' + file_extension)))


## Get paths list from idexes
def get_paths_list_from_range(folder, range_of_interest="1-", file_extension=""):
    """
    Will find the absolute path of each file with extension file_extension inside of a directory "folder",
    by following the range of interest provided.

    The formatting of range_of_interest (string) will be read as:      range_of_interest = "a-b/c"
    where files from the a-th to the b-th will be included, as well with the c-th file.

    Multiple "-" and "/" can be used. Both "a" and "b" are included, and the first element
    considered is 1 (instead of 0).

    Note: it is possible to use up to the last file available by providing a "-" at end, 
    e.g.    range_of_interest = "e-f/g-"
    will consider from e-th to f-th and from g-th until the last available file.
    """
    # Get the paths to all files of extension file_extension inside of "folder"
    full_list_of_paths = retrieve_filepaths(folder, file_extension)

    # Get the file ranges (it will be a list of lists, where there will be a list for each sub-range, each with 2 elements: start and end of the sub-range)
    files_ranges = [sub_range.split('-') for sub_range in range_of_interest.split('/')]

    # Get the indexes of interest for the corresponding files
    number_of_files = len(full_list_of_paths)
    indexes_of_interest = []
    for idx, sub_range in enumerate(files_ranges) :
        if len(sub_range) > 1 :     # an actual sub-range
            # Start of sub-range
            start = int(sub_range[0])
            # Account for cases where an end is missing
            if sub_range[1] == "":
                if idx == len(files_ranges)-1:
                    # Include files until the last one
                    end = number_of_files
                else :
                    # To cover for typos in previous sub-ranges, e.g. 1-/11-20 (take 1st, and from 11th to 20th)
                    end = start
            else :
                # End of sub-range
                end = int(sub_range[1])
            # Include indexes (1st element goes to index 0 and so on)
            sub_range_indexes = range(start-1,end)
            indexes_of_interest.extend(sub_range_indexes)
        else:   # a singular file
            file_position = int(sub_range[0])
            file_index = file_position - 1
            indexes_of_interest.append(file_index)

    # Keep only the paths in range
    paths_of_interest = []
    for idx in indexes_of_interest :
        paths_of_interest.append(full_list_of_paths[idx])


    return paths_of_interest


## Bicubic upscaling of a PIL image
def get_bicubic_image(image, scale=1):
  width, height = image.size
  width = width * scale
  height = height * scale
  new_image = image.resize((width, height), Image.BICUBIC)
  return new_image


## Use Stardist to perform segmentation on image (PNG/JPG)
def stardist_segmentation(image, mode="path", model=None, pretrained_model='2D_versatile_he',
                          scale=3, prob_thresh=None, nms_thresh=None):
  # Read image
  if mode == "path":
    image_array = imageio.imread(image)
  elif mode == "image":
    image_array = np.array(image, dtype='float32')
  else:
    image_array = image

  # Make model if not provided
  if not model :
    model = StarDist2D.from_pretrained(pretrained_model)

  # Make predictions (labels)
  labels, _ = model.predict_instances(normalize(image_array), scale=scale,
                                      prob_thresh=prob_thresh, nms_thresh=nms_thresh)

  return labels


## Use Stardist to get metrics between 2 sets of segmentation labels
def stardist_get_metrics(y_true, y_pred, metrics_names=['mean_matched_score', 'n_true'],
                         thresh=0.5, criterion='iou', report_matches=False):
  # Get metrics
  available_metrics = ["criterion", "thresh", "fp", "tp", "fn", "precision", "recall", "accuracy",
                       "f1", "n_true", "n_pred", "mean_true_score", "mean_matched_score", "panoptic_quality"]
  metrics =  matching(y_true, y_pred, thresh=thresh, criterion=criterion, report_matches=report_matches)

  # Return desired metrics
  metrics_dict = {}
  if metrics_names == "all":
    # Return all attributes
    for name in available_metrics :
      value = getattr(metrics, name, None)
      metrics_dict[name] = value

    return metrics_dict

  else:
    # Return only the provided attributes (rounded if floats)
    for name in metrics_names :
      value = getattr(metrics, name, None)
      if isinstance(value, float):
        value = round(value, 2)
      metrics_dict[name] = value

    return metrics_dict


## Convert PIL image to array (by saving and loading)
def image_to_array(image, temp_folder="", mode="imageio"):
  """
  Function to start with a PIL Image and get its equivalent array.

  This reason of making this function like this, is that when doing it with:

    image_array = np.array(image_PIL)

  We were getting discrepancies of values while measuring PSNR. I.e., an
  image_array like above would get a PSNR different from the image_array
  defined below. The latter is then preferrable.
  """
  # Get tem folder
  if not temp_folder :
    temp_folder = os.getcwd()
  # Save temp image
  image_name = "temp_image_to_array.png"
  file_path = os.path.join(temp_folder, image_name)
  image.save(file_path)

  # Load temp image as array
  if mode == "imageio" :
    image_array = imageio.imread(file_path)
  else :
    image_array = cv2.imread(file_path)

  # Delete temp image
  try:
    # Attempt to remove the file
    os.remove(file_path)
  except OSError as e:
    # Handle errors, if any
    print(f"Error: {e}")

  return image_array


## Convert PIL image to array (by saving and loading)
def array_to_image(image_array, temp_folder="", keep_on_disk=False):
  """
  Function to start with an array and get its equivalent PIL Image,
  by saving it in disk and loading it.
  """
  # Get tem folder
  if not temp_folder :
    temp_folder = os.getcwd()
  # Save temp array
  image_array_name = "temp_array_to_image.png"
  file_path = os.path.join(temp_folder, image_array_name)
  cv2.imwrite(file_path, image_array)

  # Load temp image as array
  image = Image.open(file_path)

  # Delete temp image
  if not keep_on_disk :
    try:
      # Attempt to remove the file
      os.remove(file_path)
    except OSError as e:
      # Handle errors, if any
      print(f"Error: {e}")

  return image


## Takes all JPG/PNG LR images from a folder, upscale them with the bicubic algorithm 
# to then make measures (segmentation with Stardist) respect their HR counterparts
# and export results to .csv
def measure_image_folder_with_segmentation_bicubic(LR_image_paths_list, image_folder_HR, file_extension=".png",
                                                   scale=4, metrics="all", seg_scale=3, stardist_model=None,
                                                   csv_file_name="measures", save_to="./Measurements", 
                                                   verbose=True):
  
  image_counter = 0

  # List and dictionary to store results
  images_list = []
  metrics_dict = {"criterion":[], "thresh":[], "fp":[], "tp":[], "fn":[], "precision":[], 
                  "recall":[], "accuracy":[], "f1":[], "n_true":[], "n_pred":[], 
                  "mean_true_score":[], "mean_matched_score":[], "panoptic_quality":[]}

  # Iterate over each LR image
  for LR_image_path in LR_image_paths_list :
    # Get image name
    image_name = os.path.basename(LR_image_path)
    image_name = image_name.split(".")[0]

    # Get HR counterpart path
    hr_image_name = image_name.split("_x4")[0]
    HR_image_path = os.path.join(image_folder_HR, hr_image_name + file_extension)
    images_list.append(hr_image_name)

    # Loading the LR and HR images
    LR = Image.open(LR_image_path)
    HR_array = cv2.imread(HR_image_path)

    # Print progress if asked
    if verbose :
      print(f"[INFO] Processing {hr_image_name}. Images processed: {image_counter}/{len(LR_image_paths_list)}.")

    # Upscale the image to make the Bicubic image (and convert it to an array)
    bicubic_image = get_bicubic_image(LR, scale=scale).convert('RGB')
    bicubic_image_array = np.array(bicubic_image, dtype='float32')

    # Measurements
    HR_labels = stardist_segmentation(HR_array, mode="array", model=stardist_model, scale=seg_scale)
    bicubic_labels = stardist_segmentation(bicubic_image_array, mode="array", model=stardist_model, scale=seg_scale)
    bicubic_metrics = stardist_get_metrics(y_true=HR_labels, y_pred=bicubic_labels, metrics_names=metrics)

    # Add metrics to the dictionary
    for key, value in bicubic_metrics.items() :
      metrics_dict[key].append(value)

    # An image has been processed
    image_counter+=1

  # Export measures to a csv file
  data_to_export = pd.DataFrame()
  data_to_export["image"] = images_list

  for key, value in metrics_dict.items() :
    data_to_export[key] = value

  csv_file_path = os.path.join(save_to, f"{csv_file_name}.csv")
  data_to_export.to_csv(csv_file_path)
  print(f"[INFO] Measurements for images from {image_folder_LR} has been exported to {csv_file_path}.")

  return




###################
# CODE STARTS HERE
###################

# Setting the paths to useful directories
dir_base = os.getcwd()
dir_edsrpytorch = os.path.join(dir_base,"EDSR-PyTorch")  # ./EDSR-PyTorch
dir_src = os.path.join(dir_edsrpytorch,"src")            # ./EDSR-PyTorch/src
dir_images = os.path.join(dir_base,"images")             # ./images

# Provide images folders
dir_dataset = os.path.join(dir_edsrpytorch, "image-data-general")
#dir_dataset = os.path.join(dir_edsrpytorch, "image-data-dedicated")
image_folder_LR = os.path.join(dir_dataset, "Custom", "LR_bicubic", "X4")  # ./EDSR-PyTorch/image-data/Custom/LR_bicubic/X4
image_folder_HR = os.path.join(dir_dataset, "Custom", "HR")                # ./EDSR-PyTorch/image-data/Custom/HR
file_extension=".png"

# Get list of LR images of interest
range_of_interest = "2401-2500"
range_of_interest = "30972-32261"
#range_of_interest = "start_1-end_1/start_2-end_2/.../start_n-end_n"
LR_image_paths_list = get_paths_list_from_range(folder=image_folder_LR, range_of_interest=range_of_interest, file_extension=file_extension)

# Provide our model parameters
scale = 4

# Provide parameters for the measurements
pretrained_stardist_model = "2D_versatile_he"
metrics = "all"
seg_scale = 3

# Provide other parameters
verbose = True
csv_file_name = "bicubic-measures_with_segmentation_general_dataset_trained_general"

# Set up paths and folders
path_to_save = os.path.join(dir_base, "Measurements", "General")
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save, exist_ok=True)
current_image_folder = os.path.join(dir_images, "current_image_measuring")  # ./images/current_image_measuring


# Make Stardist Model
stardist_model = StarDist2D.from_pretrained(pretrained_stardist_model)


# Measure folder
measure_image_folder_with_segmentation_bicubic(LR_image_paths_list, image_folder_HR, file_extension=file_extension,
                                               scale=scale, metrics=metrics, seg_scale=seg_scale, stardist_model=stardist_model,
                                               csv_file_name=csv_file_name, save_to=path_to_save, verbose=True)