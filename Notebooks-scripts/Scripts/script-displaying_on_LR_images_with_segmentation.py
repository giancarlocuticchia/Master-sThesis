import cv2
import glob
from PIL import Image
import imageio
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import subprocess

import stardist
from stardist.models import StarDist2D
from csbdeep.utils import normalize
from stardist.matching import matching
from stardist.plot import render_label


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


## Make composition of showing a LR image and its upscaled versions (Bicubic+EDSR)
# from a list of image files paths, with an associated segmentation
def make_composition_of_low_res_with_segmentation(list_of_images_paths, edsr_pretrain_path="", edsr_name="", scale=4,
                                                  seg_scale=3, force_color=None, metrics_name="tp",  stardist_model=None,
                                                  comp_name="composition_from_sample", save_to="./CompositionFromSample",
                                                  current_image_folder="./EDSR-PyTorch/test", dir_src="./EDSR-PyTorch/src",
                                                  verbose=True, fig_size=(10,10), fontsize=8, show_fig=True):
  """
  Will take a list of image files paths (list_of_images_paths), e.g. content/image6.png,
  load each of them and upscale them using Bicubic and EDSR algorithm with the
  provided parameters.

  Will perform segmentation using Stardist with the parameters provided (e.g. seg_scale),
  showing the segmentation for each image next to it, and showing the metric
  provided in metrics_name by using as the true values the segmentation labels
  obtained from the upscaled image with itself.

  The final composition will have a row for each image from the list and a column for the
  original (low resolution) version, bicubic upscale, EDSR upscale, and a segmentation for both.

  Will save the final composition with name comp_name in the path save_to.
  """
  # Make figure and axes
  number_of_rows = len(list_of_images_paths)     # One for each image
  number_of_columns = 5                           # Original, Bicubic + Segmentation, and EDSR + Segmentation
  fig, axs = plt.subplots(number_of_rows, number_of_columns, figsize=fig_size)

  # Iterate over each image from the sample
  for row, image_filepath in enumerate(list_of_images_paths):
    # Get image name
    image_filename = os.path.basename(image_filepath)
    image_name = image_filename.split('.')[0]

    # Loading LR image
    LR_image = Image.open(image_filepath)        #Alternatively:   LR = mpimg.imread(image_filepath)    where:  import matplotlib.image as mpimg

    # Get dimensions of LR image
    LR_image_width, LR_image_height = LR_image.size
    # Create a white background image of double the size
    background_width = 2 * LR_image_width
    background_height = 2 * LR_image_height
    background = Image.new('RGB', (background_width, background_height), (255, 255, 255))
    # Calculate position to paste LR image on background to center it
    paste_position = ((background_width - LR_image_width) // 2, (background_height - LR_image_height) // 2)
    # Paste LR image onto the background
    background.paste(LR_image, paste_position)

    # Plot the original LR image for the current row
    axs[row, 0].imshow(background, aspect='equal')
    axs[row, 0].axis('off')
    axs[row, 0].set_title(f"Original\n{image_name}\n", fontsize=fontsize)


    # Plot the bicubic upscaled image for the LR image of the current row
    bicubic_image = get_bicubic_image(LR_image, scale=scale).convert('RGB')
    bicubic_image_array = np.array(bicubic_image, dtype='float32')
    axs[row, 1].imshow(bicubic_image)
    axs[row, 1].axis('off')
    axs[row, 1].set_title(f"Bicubic\n", fontsize=fontsize)
    # Get the segmentation for the original image
    bicubic_labels = stardist_segmentation(bicubic_image_array, mode="array", model=stardist_model, scale=seg_scale)
    bicubic_metrics = stardist_get_metrics(y_true=bicubic_labels, y_pred=bicubic_labels, metrics_names="all")
    bicubic_title = f"{metrics_name}: {bicubic_metrics[metrics_name]}\n"
    bicubic_image_rendered = render_label(bicubic_labels, img=bicubic_image_array, normalize_img = False, alpha=0.75, alpha_boundary=1)
    if force_color :
      bicubic_image_rendered[bicubic_labels > 0] = force_color
    axs[row, 2].imshow(np.uint8(bicubic_image_rendered[..., :3]))
    axs[row, 2].axis('off')
    axs[row, 2].set_title(bicubic_title, fontsize=fontsize)


    # Plot the upscaled image for EDSR
    if verbose :
      print(f"[INFO] Applying EDSR from {edsr_pretrain_path} to image {image_filename} in directory {dir_src}.")
    # Apply EDSR model
    SR_array = apply_EDSR_to_image(LR_image, folder=current_image_folder, data_test="Demo",
                                   scale=scale, pretrain_path=edsr_pretrain_path,
                                   dir_src=dir_src, verbose=False, keep_on_disk=False)
    # Plot it
    SR_image = array_to_image(SR_array, temp_folder=save_to)
    SR_array = image_to_array(SR_image, temp_folder=save_to, mode="imageio")
    axs[row, 3].imshow(SR_image)
    axs[row, 3].axis('off')
    axs[row, 3].set_title(f"{edsr_name}\n", fontsize=fontsize)
    # Get the segmentation for the SR image
    SR_labels = stardist_segmentation(SR_array, mode="array", model=stardist_model, scale=seg_scale)
    SR_metrics = stardist_get_metrics(y_true=SR_labels, y_pred=SR_labels, metrics_names="all")
    SR_title = f"{metrics_name}: {SR_metrics[metrics_name]}\n"
    SR_image_rendered = render_label(SR_labels, img=SR_array, normalize_img = False, alpha=0.75, alpha_boundary=1)
    if force_color :
      SR_image_rendered[SR_labels > 0] = force_color
    axs[row, 4].imshow(np.uint8(SR_image_rendered[..., :3]))
    axs[row, 4].axis('off')
    axs[row, 4].set_title(SR_title, fontsize=fontsize)


  # Adjust layout and show
  plt.tight_layout()
  if show_fig:
    plt.show()

  # Save composition on disk
  if not os.path.exists(save_to):
    os.makedirs(save_to, exist_ok=True)

  composition_name = f"{comp_name}.png"
  export_path = os.path.join(save_to,composition_name)
  fig.savefig(export_path)
  if verbose :
    print(f"Composition {composition_name} was successfully saved in {save_to}.\n")

  # Close the figure
  plt.close()

  return


## Retrieve a sorted list of the paths of all the files with a certain extension inside a directory
def retrieve_filepaths(directory, file_extension=""):
  return sorted(glob.glob(os.path.join(directory, '*' + file_extension)))






###################
# CODE STARTS HERE
###################

# Setting the paths to useful directories
dir_base = os.getcwd()
dir_edsrpytorch = os.path.join(dir_base,"EDSR-PyTorch")  # ./EDSR-PyTorch
dir_src = os.path.join(dir_edsrpytorch,"src")            # ./EDSR-PyTorch/src
dir_images = os.path.join(dir_base,"images")             # ./images

# Set up paths and folders
path_to_save = os.path.join(dir_images, "CompositionSegmentationLR")            # ./images/CompositionSegmentationLR
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save, exist_ok=True)
current_image_folder = os.path.join(dir_images, "current_image_display-segLR")  # ./images/current_image_display-segLR

# Provide images folders
file_extension=".png"
#image_folder = path_to_the_folder_with_our_LR_images
image_folder = dir_images    # Example

# Model parameters
scale = 4
dir_pretrained = os.path.join(dir_edsrpytorch, "pre-train")
#edsr_pretrain_path = os.path.join(dir_pretrained, "edsr_x4-4f62e9ef.pt")                            # pre-trained EDSR (by authors)
#edsr_pretrain_path = os.path.join(dir_pretrained, "edsr_x4-best_trained_general.pt")                # trained EDSR with general dataset (TCIA)
edsr_pretrain_path = os.path.join(dir_pretrained, "edsr_x4-best_trained_dedicated_whole.pt")        # trained EDSR with dedicated dataset (Humanitas), whole dataset
#edsr_pretrain_path = os.path.join(dir_pretrained, "edsr_x4-best_trained_dedicated_inbatches.pt")    # trained EDSR with dedicated dataset (Humanitas), dataset in batches

edsr_name = "EDSR"

# Other parameters
pretrained_stardist_model = "2D_versatile_he"
seg_scale = 3
#force_color = None
force_color = [0, 255, 0, 1]
metrics_name = "tp"
verbose = False
verbose_comp = True

# Figure parameters (for a dynamic fig_size keep reading code below)
#fig_size = (10, 20)
fontsize = 12
show_fig = False




###################
# Get list of image files in the folder
list_of_images_paths = retrieve_filepaths(image_folder, file_extension=file_extension)
n_images = len(list_of_images_paths)

# Make Stardist Model
stardist_model = StarDist2D.from_pretrained(pretrained_stardist_model)

# Get fig_size
images_width = 256
images_height = 256
fig_size = (5 * images_width / 100, n_images * images_height / 100)

# Make composition from sample
comp_name = f"composition_LR_images_with_segmentation"

make_composition_of_low_res_with_segmentation(list_of_images_paths=list_of_images_paths,
                                              edsr_pretrain_path=edsr_pretrain_path, edsr_name=edsr_name, scale=scale,
                                              seg_scale=seg_scale, force_color=force_color, metrics_name=metrics_name,
                                              stardist_model=stardist_model, comp_name=comp_name, save_to=path_to_save,
                                              current_image_folder=current_image_folder, dir_src=dir_src,
                                              verbose=verbose_comp, fig_size=fig_size, fontsize=fontsize, show_fig=show_fig)

if verbose :
    print(f"Composition of LR images with segmentation has been saved as {comp_name} in {path_to_save}.")