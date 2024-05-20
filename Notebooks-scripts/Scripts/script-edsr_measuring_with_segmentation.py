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

## Apply EDSR model to all JPG/PNG images inside folder
#NOTE: Make sure images are RGB and not RGBA
def apply_EDSR_to_folder(folder="./EDSR-PyTorch/test",
                         data_test="Demo", scale=4, pretrain_path="",
                         dir_src="./EDSR-PyTorch/src", verbose=True):
  ## Change the current working directory to EDSR-PyTorch/src, to run the model
  starting_dir = os.getcwd()
  current_dir = starting_dir
  # Check if our directory is ./EDSR-PyTorch/src
  if current_dir.endswith("EDSR-PyTorch/src"):
    if verbose :
      print(f"[INFO] The Current working directory is: {current_dir}\n")
  else:
    # Changing the current working directory
    os.chdir(dir_src)
    if verbose:
      print(f"[INFO] The Current working directory is: {os.getcwd()}\n")

  ## Use the model
  if verbose :
    print(f"[INFO] Using EDSR_x{scale} model as {data_test} with weights from {pretrain_path} on images inside {folder}.\n")
  command = f"python main_use.py --data_test {data_test} --dir_demo {folder} --scale {scale} --save test --n_resblocks 32 --n_feats 256 --res_scale 0.1 --pre_train {pretrain_path} --test_only --save_results"
  subprocess.run(command, shell=True)

  # Change the current working directory back to the previous one
  os.chdir(starting_dir)
  if verbose :
    print(f"[INFO] The Current working directory is: {starting_dir}\n")

  return


## Apply EDSR to a single image (returned image is an array)
def apply_EDSR_to_image(image, folder="./EDSR-PyTorch/test",
                        data_test="Demo", scale=4, pretrain_path="",
                        dir_src="./EDSR-PyTorch/src", verbose=True,
                        keep_on_disk=False):
  # Make copy of image for manipulation (without modifying the original one)
  input_image = image.copy()

  # If image not RGB, make it RGB
  if input_image.mode != "RGB":
    input_image = input_image.convert("RGB")

  # Save image on disk
  if not os.path.exists(folder):
    os.makedirs(folder, exist_ok=True)
  image_path = os.path.join(folder,"current_image.png")
  input_image.save(image_path)

  # Use EDSR model
  apply_EDSR_to_folder(folder=folder, data_test=data_test, scale=scale,
                       pretrain_path=pretrain_path, dir_src=dir_src, verbose=verbose)

  # Read upscaled image
  upscaled_image_filename = f"current_image_x{scale}.png"
  upscaled_image_path = os.path.join(folder,"results",upscaled_image_filename)
  upscaled_image = cv2.imread(upscaled_image_path)

  if not keep_on_disk :
    # Delete created directories and files
    try:
      # Attempt to remove the directory and its contents
      shutil.rmtree(folder)
    except OSError as e:
      # Handle errors, if any
      print(f"Error: {e}")

  # Return upscaled image
  return upscaled_image


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


## Takes all JPG/PNG LR images from a folder, upscale them with the EDSR (provided 
# from pretrain_path), to then make measures (segmentation with Stardist) respect 
# their HR counterparts and export results to .csv
def measure_image_folder_with_segmentation(LR_image_paths_list, image_folder_HR, file_extension=".png",
                                           pretrain_path="download", scale=4, 
                                           metrics="all", seg_scale=3, stardist_model=None,
                                           csv_file_name="measures", save_to="./Measurements", 
                                           current_image_folder="./EDSR-PyTorch/test",
                                           dir_src="./EDSR-PyTorch/src", verbose=True):
  
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

    # Upscale the image to make the SR image (as array)
    SR_array = apply_EDSR_to_image(LR, folder=current_image_folder, data_test="Demo", 
                                   scale=scale, pretrain_path=pretrain_path,
                                   dir_src=dir_src, verbose=False, keep_on_disk=False)

    # Measurements
    HR_labels = stardist_segmentation(HR_array, mode="array", model=stardist_model, scale=seg_scale)
    SR_labels = stardist_segmentation(SR_array, mode="array", model=stardist_model, scale=seg_scale)
    SR_metrics = stardist_get_metrics(y_true=HR_labels, y_pred=SR_labels, metrics_names=metrics)

    # Add metrics to the dictionary
    for key, value in SR_metrics.items() :
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
dir_pretrained = os.path.join(dir_edsrpytorch, "pre-train")
#pretrain_path = os.path.join(dir_pretrained, "edsr_x4-4f62e9ef.pt")                             # pre-trained EDSR (by authors)
pretrain_path = os.path.join(dir_pretrained, "edsr_x4-best_trained_general.pt")                 # trained EDSR with general dataset (TCIA)
#pretrain_path = os.path.join(dir_pretrained, "edsr_x4-best_trained_dedicated_whole.pt")         # trained EDSR with dedicated dataset (Humanitas), whole dataset
#pretrain_path = os.path.join(dir_pretrained, "edsr_x4-best_trained_dedicated_inbatches.pt")     # trained EDSR with dedicated dataset (Humanitas), dataset in batches

# Provide parameters for the measurements
pretrained_stardist_model = "2D_versatile_he"
metrics = "all"
seg_scale = 3

# Provide other parameters
verbose = True
csv_file_name = "measures_with_segmentation_general_dataset_trained_general"


# Set up paths and folders
path_to_save = os.path.join(dir_base, "Measurements", "General")
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save, exist_ok=True)
current_image_folder = os.path.join(dir_images, "current_image_measuring_seg")  # ./images/current_image_measuring_seg


# Make Stardist Model
stardist_model = StarDist2D.from_pretrained(pretrained_stardist_model)


# Measure folder
measure_image_folder_with_segmentation(LR_image_paths_list, image_folder_HR, file_extension=file_extension,
                                       pretrain_path=pretrain_path, scale=scale, 
                                       metrics=metrics, seg_scale=seg_scale, stardist_model=stardist_model,
                                       csv_file_name=csv_file_name, save_to=path_to_save, 
                                       current_image_folder=current_image_folder, dir_src=dir_src, verbose=verbose)