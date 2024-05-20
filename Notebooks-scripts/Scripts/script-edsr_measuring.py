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


## Retrieve a sorted list of the paths of all the files with a certain extension inside a directory
def retrieve_filepaths(directory, file_extension=""):
  return sorted(glob.glob(os.path.join(directory, '*' + file_extension)))


## Returns a list of paths for all the files with a certain extension inside a directory and its subdirectories
def get_filepaths(directory, file_extension=""):
    files_list = []
    # Walk through the directory and its subdirectories
    for dir, subdirs, files in os.walk(directory):
        for file in files:
            # Check if the file has a the extension
            if file.endswith(file_extension):
                # Add the full path to the list
                files_list.append(os.path.join(dir, file))

    return files_list


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

## Mean Squared Error between two images
def calc_mse(imageA, imageB):
  """
  'Mean Squared Error' between two images of same dimensions and type (RGB, for example)

  Images must be an array (e.g. Array of iunt8 of shape (height,width,3))

  MSE = sum over N of (Ai - Bi)^2 / N     where N is the number of pixels per image

  Images are converted to float32 precision.
  """
  imageA_float32 = imageA.astype(np.float32)
  imageB_float32 = imageB.astype(np.float32)

  MSE = np.sum((imageA_float32 - imageB_float32)**2)
  MSE /= math.prod(imageA_float32.shape[dim] for dim in range(len(imageA_float32.shape)))

  return MSE


## Peak-Signal-To-Noise Ratio between a SR and a HR image
def calc_psnr_simplified(sr, hr, scale, rgb_range=255, crop=True):
  """
  Notes: 
    * sr and hr must be a numpy array.
    * sr and hr are converted to float32 precision.
    * sr and hr might be cropped from their borders with a number "shave" of
    pixels, if crop=True.
  """
  diff = (sr.astype(np.float32) - hr.astype(np.float32))  # Ensure float32 precision
  if crop :
    shave = scale + 6
    valid = diff[shave:-shave, shave:-shave, ...]         # Cropping out borders of the image
  else :
    valid = diff
  mse = np.mean(valid ** 2)
  psnr = -10 * math.log10(mse / (rgb_range ** 2))

  return psnr


## Structural Similarity between two images
def calc_ssim(imageA, imageB, channel_axis=2):
  """
  'Structural Similarity' between two images, calculated with "structural_similarity"
  from the library skimage.metrics.

  Images must be an array (e.g. Array of iunt8 of shape (height,width,3)) and
  are converted to float32 precision.

  The argument channel_axis correspond to the index of the arrays corresponding
  to their color channels (typically 0 or 2).
  """
  imageA_float32 = imageA.astype(np.float32)
  imageB_float32 = imageB.astype(np.float32)

  SSIM = skimage_ssim(imageA_float32, imageB_float32, channel_axis=channel_axis)

  return SSIM


## Takes all JPG/PNG LR images from a folder, upscale them with the EDSR (provided 
# from pretrain_path), to then make measures (PSNR, MSE or SSIM) respect their HR
# counterparts and export results to .csv
def measure_image_folder(LR_image_paths_list, image_folder_HR, file_extension=".png",
                         pretrain_path="download", measures=["PSNR"], scale=4,
                         rgb_range=255, crop=True, csv_file_name="measures",
                         save_to="./Measurements", current_image_folder="./EDSR-PyTorch/test",
                         dir_src="./EDSR-PyTorch/src", verbose=True):
  
  image_counter = 0

  # Lists to store results
  images_list = []
  MSE_list = []
  PSNR_list = []
  SSIM_list = []

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
    for measure in measures :
      if measure == "MSE":
        #Mean Squared Error
        MSE = calc_mse(HR_array,SR_array)
        MSE_list.append(MSE)

      if measure == "PSNR":
        #Peak Signal-to-Noise Ratio
        PSNR = calc_psnr_simplified(SR_array, HR_array, scale=scale, rgb_range=rgb_range, crop=crop)
        PSNR_list.append(PSNR)

      if measure == "SSIM":
        #Structural Similarity
        SSIM = calc_ssim(HR_array,SR_array,channel_axis=2)
        SSIM_list.append(SSIM)

    #An image has been processed
    image_counter+=1

  # Export measures to a csv file
  data_to_export = pd.DataFrame()
  data_to_export["image"] = images_list

  for measure in measures :
    if measure == "MSE": data_to_export["MSE"] = MSE_list
    if measure == "PSNR": data_to_export["PSNR"] = PSNR_list
    if measure == "SSIM": data_to_export["SSIM"] = SSIM_list

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
measures = ["MSE", "PSNR", "SSIM"]
rgb_range = 255
crop = True

# Provide other parameters
verbose = True
csv_file_name = "measures_general_dataset_trained_general"


# Set up paths and folders
path_to_save = os.path.join(dir_base, "Measurements", "General")
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save, exist_ok=True)
current_image_folder = os.path.join(dir_images, "current_image_measuring")  # ./images/current_image_measuring

# Measure folder
measure_image_folder(LR_image_paths_list, image_folder_HR, file_extension=file_extension,
                     pretrain_path=pretrain_path, measures=measures, scale=scale,
                     rgb_range=rgb_range, crop=crop, csv_file_name=csv_file_name,
                     save_to=path_to_save, current_image_folder=current_image_folder,
                     dir_src=dir_src, verbose=verbose)