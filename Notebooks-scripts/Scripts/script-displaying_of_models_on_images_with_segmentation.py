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
def stardist_segmentation(image, mode="path", pretrained_model='2D_versatile_he',
                          scale=3, prob_thresh=None, nms_thresh=None):
  # Read image
  if mode == "path":
    image_array = imageio.imread(image)
  elif mode == "image":
    image_array = np.array(image, dtype='float32')
  else:
    image_array = image

  # Make model
  model = StarDist2D.from_pretrained(pretrained_model)

  # Make predictions (labels)
  labels, _ = model.predict_instances(normalize(image_array), scale=scale,
                                      prob_thresh=prob_thresh, nms_thresh=nms_thresh)

  return labels


## Use Stardist to get metrics between 2 sets of segmentation labels
def stardist_get_metrics(y_true, y_pred, metrics_names=['mean_matched_score', 'n_true'],
                         thresh=0.5, criterion='iou', report_matches=False):
  # Get metrics
  metrics =  matching(y_true, y_pred, thresh=thresh, criterion=criterion, report_matches=report_matches)

  # Return desired metrics
  if metrics_names == "all":
    return metrics

  else:
    metrics_dict = {}
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


## Make composition of showing a HR image and the upscaled versions of its LR counterpart from a sample of images
# and showing segmentation and associated metrics
def make_composition_from_sample_with_segmentation(list_of_sample_images, image_folder_HR, image_folder_LR,
                                                   list_of_edsr_pretrain_paths="", list_of_edsr_names="",
                                                   scale=4, seg_scale=3, force_color=None,
                                                   metrics_names=['mean_matched_score', 'n_true'],
                                                   comp_name="composition_from_sample", save_to="./CompositionFromSample",
                                                   current_image_folder="./EDSR-PyTorch/test", dir_src="./EDSR-PyTorch/src",
                                                   verbose=True, fig_size=(10,10), fontsize=8, show_fig=True):
  """
  Will take a list of images filenames (list_of_sample_images), e.g. image6.png,
  and load it from its folder (image_folder_HR) and will also load its low resolution
  version (from image_folder_LR) with its appropiated name, e.g. image6_x4.png.

  Will take the LR version of the image and upscale it (according to scale, e.g.
  4 times) with a bicubic algorithm and by using the EDSR model for each pretrain
  path provided (list_of_edsr_pretrain_paths). For each pretrain path, there must
  be its corresponding name or label (list_of_edsr_names).
  Note: The length of list_of_edsr_pretrain_paths and list_of_edsr_names must match.

  Will perform segmentation using Stardist with the parameters provided (e.g. seg_scale),
  showing the segmentation for each image below it, and showing the metrics
  provided in metrics_names by using as the true values the segmentation labels
  obtained from the original HR image.

  The final composition will have two rows for each image from the sample (image
  and segmentation) and a column for the original version, bicubic upscale and each
  EDSR upscaling.

  Will save the final composition with name comp_name in the path save_to.
  """
  # Make figure and axes
  number_of_rows = 2*len(list_of_sample_images)               # One set for the images, one set for the segmentations
  number_of_columns = 2 + len(list_of_edsr_pretrain_paths)    # Original, Bicubic, and each EDSR
  fig, axs = plt.subplots(number_of_rows, number_of_columns, figsize=fig_size)

  # Iterate over each image from the sample
  for row, image_filename in enumerate(list_of_sample_images):
    # Index values for img and segmentation
    row_img_idx = 2*row
    row_seg_idx = 2*row + 1

    # Get image name
    image_name = image_filename.split('.')[0]

    # Get HR image path
    HR_image_path = os.path.join(image_folder_HR, image_filename)

    # Get LR image path
    LR_image_filename = f"{image_name}_x4.{image_filename.split('.')[-1]}"    #Using the same file extension as the HR counterpart
    LR_image_path = os.path.join(image_folder_LR, LR_image_filename)

    # Loading LR and HR images
    LR_image = Image.open(LR_image_path)        #Alternatively:   LR = mpimg.imread(LR_image_path)    where:  import matplotlib.image as mpimg
    HR_image = imageio.imread(HR_image_path)


    # Plot the original image for the current row
    axs[row_img_idx, 0].imshow(HR_image)
    axs[row_img_idx, 0].axis('off')
    axs[row_img_idx, 0].set_title(f"Original\n{image_name}\n", fontsize=fontsize)
    # Get the segmentation for the original image
    HR_labels = stardist_segmentation(HR_image, mode="image", scale=seg_scale)
    HR_metrics = stardist_get_metrics(y_true=HR_labels, y_pred=HR_labels, metrics_names=metrics_names)
    HR_title = f"segmentation\n{metrics_names[-1]}: {HR_metrics[metrics_names[-1]]}"
    HR_image_rendered = render_label(HR_labels, img=HR_image, normalize_img = False, alpha=0.75, alpha_boundary=1)
    if force_color :
      HR_image_rendered[HR_labels > 0] = force_color
    axs[row_seg_idx, 0].imshow(np.uint8(HR_image_rendered[..., :3]))
    axs[row_seg_idx, 0].axis('off')
    axs[row_seg_idx, 0].set_title(HR_title, fontsize=fontsize)


    # Plot the bicubic upscaled image for the LR image of the current row
    bicubic_image = get_bicubic_image(LR_image, scale=scale)
    bicubic_image_array = np.array(bicubic_image, dtype='float32')
    axs[row_img_idx, 1].imshow(bicubic_image)
    axs[row_img_idx, 1].axis('off')
    axs[row_img_idx, 1].set_title(f"Bicubic\n", fontsize=fontsize)
    # Get the segmentation for the original image
    bicubic_labels = stardist_segmentation(bicubic_image_array, mode="array", scale=seg_scale)
    bicubic_metrics = stardist_get_metrics(y_true=HR_labels, y_pred=bicubic_labels, metrics_names=metrics_names)
    bicubic_title = '\n'.join([f"{key}: {value}" for key, value in bicubic_metrics.items()])
    bicubic_image_rendered = render_label(bicubic_labels, img=bicubic_image_array, normalize_img = False, alpha=0.75, alpha_boundary=1)
    if force_color :
      bicubic_image_rendered[bicubic_labels > 0] = force_color
    axs[row_seg_idx, 1].imshow(np.uint8(bicubic_image_rendered[..., :3]))
    axs[row_seg_idx, 1].axis('off')
    axs[row_seg_idx, 1].set_title(bicubic_title, fontsize=fontsize)


    # Plot the upscaled image for each different EDSR
    for edsr_idx, pretrain_path in enumerate(list_of_edsr_pretrain_paths):
      if verbose :
        print(f"[INFO] Applying EDSR from {pretrain_path} to image {image_filename} in directory {dir_src}.")
      # Apply EDSR model
      SR_array = apply_EDSR_to_image(LR_image, folder=current_image_folder, data_test="Demo",
                                     scale=scale, pretrain_path=pretrain_path,
                                     dir_src=dir_src, verbose=False, keep_on_disk=False)
      # Plot it
      #SR_image = Image.fromarray(np.uint8(SR_array)).convert('RGB')      # This line doesn't work (colors affected)
      SR_image = array_to_image(SR_array, temp_folder=save_to)
      SR_array = image_to_array(SR_image, temp_folder=save_to, mode="imageio")
      edsr_name = list_of_edsr_names[edsr_idx]
      axs[row_img_idx, edsr_idx+2].imshow(SR_image)
      axs[row_img_idx, edsr_idx+2].axis('off')
      axs[row_img_idx, edsr_idx+2].set_title(f"{edsr_name}\n", fontsize=fontsize)
      # Get the segmentation for the SR image
      SR_labels = stardist_segmentation(SR_array, mode="array", scale=seg_scale)
      SR_metrics = stardist_get_metrics(y_true=HR_labels, y_pred=SR_labels, metrics_names=metrics_names)
      SR_title = '\n'.join([f"{key}: {value}" for key, value in SR_metrics.items()])
      SR_image_rendered = render_label(SR_labels, img=SR_array, normalize_img = False, alpha=0.75, alpha_boundary=1)
      if force_color :
        SR_image_rendered[SR_labels > 0] = force_color
      axs[row_seg_idx, edsr_idx+2].imshow(np.uint8(SR_image_rendered[..., :3]))
      axs[row_seg_idx, edsr_idx+2].axis('off')
      axs[row_seg_idx, edsr_idx+2].set_title(SR_title, fontsize=fontsize)


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


## Function to make a number have the same number of digits that a reference
def make_numbers_the_same_length(target, reference):
    """
    Make the number "target" have the same number of digits as the number "reference".

    Returns a string.
    """
    target_str = str(target)
    reference_str = str(reference)

    # Compare the lengths of the numbers
    if len(target_str) < len(reference_str):
        # Pad the target with zeros on the left
        target_str = target_str.zfill(len(reference_str))

    return target_str


## Function to check if both dimensions of an image are multiple of a number
def are_image_dimensions_multiple_of(image_path, multiple_of=1):
    """
    Check if both dimensions of the image are multiples of a number (equal to multiple_of).
    """
    img = Image.open(image_path)
    width, height = img.size
    return width % multiple_of == 0 and height % multiple_of == 0

## Function to pick a random sample of images from a list, if both image dimensions are multiple of a number
def select_random_images_if_multiple_of(list_of_images_paths, num_images=5, multiple_of=1):
    """
    Will take a list of paths to images, and take a random sample of them (equal
    to num_images) only if both dimensions (width, height) are multiple of a
    number n (equal to multiple_of), and return a list with the images filenames,
    e.g., [image1.png, image2.png, image3.png, image4.png, image5.png]
    """
    # Filter images with both dimensions being multiples of the number
    valid_images = [image_path for image_path in list_of_images_paths if are_image_dimensions_multiple_of(image_path, multiple_of=multiple_of)]

    # Check if there are enough valid images
    if len(valid_images) < num_images:
        print(f"[Error] Not enough images with both dimensions multiples of {multiple_of} in the provided list.")
        return None

    # Keep only the images filenames
    valid_images = [os.path.basename(path) for path in valid_images]

    # Select random sample from the valid images
    random_images = random.sample(valid_images, num_images)

    return random_images






###################
# CODE STARTS HERE
###################

# Setting the paths to useful directories
dir_base = os.getcwd()
dir_edsrpytorch = os.path.join(dir_base,"EDSR-PyTorch")  # ./EDSR-PyTorch
dir_src = os.path.join(dir_edsrpytorch,"src")            # ./EDSR-PyTorch/src
dir_images = os.path.join(dir_base,"images")             # ./images

# Set up paths and folders
path_to_save = os.path.join(dir_images, "CompositionSegmentationFromSamples")       # ./images/CompositionSegmentationFromSamples
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save, exist_ok=True)
current_image_folder = os.path.join(dir_images, "current_image_display-seg")        # ./images/current_image_display-seg

# Provide images folders
dir_dataset = os.path.join(dir_edsrpytorch, "image-data-general")
#dir_dataset = os.path.join(dir_edsrpytorch, "image-data-dedicated")
image_folder_LR = os.path.join(dir_dataset, "Custom", "LR_bicubic", "X4")  # ./EDSR-PyTorch/image-data/Custom/LR_bicubic/X4
image_folder_HR = os.path.join(dir_dataset, "Custom", "HR")                # ./EDSR-PyTorch/image-data/Custom/HR
file_extension=".png"

# Model parameters
scale = 4
dir_pretrained = os.path.join(dir_edsrpytorch, "pre-train")
list_of_edsr_pretrain_paths = [
    os.path.join(dir_pretrained, "edsr_x4-4f62e9ef.pt"),                            # pre-trained EDSR (by authors)
    os.path.join(dir_pretrained, "edsr_x4-best_trained_general.pt"),                # trained EDSR with general dataset (TCIA)
    os.path.join(dir_pretrained, "edsr_x4-best_trained_dedicated_whole.pt"),        # trained EDSR with dedicated dataset (Humanitas), whole dataset
    os.path.join(dir_pretrained, "edsr_x4-best_trained_dedicated_inbatches.pt")     # trained EDSR with dedicated dataset (Humanitas), dataset in batches
    ]
list_of_edsr_names = ["Pretrained EDSR", "Trained on general dataset", "On whole dedicated dataset", "On dedicated dataset in batches"]

# Other parameters
seg_scale = 3
force_color = None
force_color = [0, 255, 0, 1]
metrics_names = ['mean_matched_score', 'tp']
verbose = False
verbose_comp = True
start_image = 30972
end_image = 32261
number_of_samples = 5     # Number of images to take for the sample

# Figure parameters
#fig_size = (10, 20)
#fig_size = (15, 5)
fig_size = (15, 5 * number_of_samples)
fontsize = 8
show_fig = False




###################
# Get list of image files in the folder
list_of_images_paths = retrieve_filepaths(image_folder_HR, file_extension=file_extension)
n_images = len(list_of_images_paths)

# Restrict the images to the corresponding range
list_of_images_paths_in_range = list_of_images_paths[start_image-1:end_image]

# Select a random sample from the range that can be used for comparison (i.e., their dimensions (width, height) are multiple of the scale of the model)
sample_of_images = select_random_images_if_multiple_of(list_of_images_paths_in_range,
                                                       num_images=number_of_samples, multiple_of=scale)

# Make composition from sample
range_start_formatted = make_numbers_the_same_length(start_image, n_images)
range_end_formatted = make_numbers_the_same_length(end_image, n_images)
comp_name = f"composition_from_sample_with_segmentation_in_range{range_start_formatted}-{range_end_formatted}"

make_composition_from_sample_with_segmentation(list_of_sample_images=sample_of_images,
                                               image_folder_HR=image_folder_HR, image_folder_LR=image_folder_LR,
                                               list_of_edsr_pretrain_paths=list_of_edsr_pretrain_paths,
                                               list_of_edsr_names=list_of_edsr_names, scale=scale,
                                               seg_scale=seg_scale, force_color=force_color, metrics_names=metrics_names,
                                               comp_name=comp_name, save_to=path_to_save,
                                               current_image_folder=current_image_folder,
                                               dir_src=dir_src, verbose=verbose_comp,
                                               fig_size=fig_size, fontsize=fontsize, show_fig=show_fig)


if verbose :
    print(f"Composition with segmentation for range {range_start_formatted}-{range_end_formatted} has been saved as {comp_name} in {path_to_save}.")

