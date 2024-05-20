import cv2
import glob
from PIL import Image
import math
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
from skimage.metrics import structural_similarity as skimage_ssim
import subprocess


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
  try :
    psnr = -10 * math.log10(mse / (rgb_range ** 2))
  except :
    psnr = float('nan')

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


## Convert PIL image to array (by saving and loading)
def image_to_array(image, temp_folder=""):
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
def make_composition_from_sample(list_of_sample_images, image_folder_HR, image_folder_LR,
                                 list_of_edsr_pretrain_paths="", list_of_edsr_names="",
                                 scale=4, rgb_range=255, crop=True,
                                 comp_name="composition_from_sample", save_to="./CompositionFromSample",
                                 current_image_folder="./EDSR-PyTorch/test",
                                 dir_src="./EDSR-PyTorch/src", verbose=True,
                                 fig_size=(10,10), fontsize=8, show_fig=True):
  """
  Will take a list of images filenames (list_of_sample_images), e.g. image6.png,
  and load it from its folder (image_folder_HR) and will also load its low resolution
  version (from image_folder_LR) with its appropiated name, e.g. image6_x4.png.

  Will take the LR version of the image and upscale it (according to scale, e.g. 
  4 times) with a bicubic algorithm and by using the EDSR model for each pretrain
  path provided (list_of_edsr_pretrain_paths). For each pretrain path, there must
  be its corresponding name or label (list_of_edsr_names).
  Note: The length of list_of_edsr_pretrain_paths and list_of_edsr_names must match.

  Will measure MSE, PSNR and SSIM of each upscaled version of the image with respect
  the original one, and show its corresponding values as well as the upscaled
  images on a final composition, where there will be a row for each image from
  the sample and a column for the original version, bicubic upscale and each EDSR
  upscaling.

  Will save the final composition with name comp_name in the path save_to.
  """
  # Make figure and axes
  number_of_rows = len(list_of_sample_images)
  number_of_columns = 2 + len(list_of_edsr_pretrain_paths)    # Original, Bicubic, and each EDSR
  fig, axs = plt.subplots(number_of_rows, number_of_columns, figsize=fig_size)

  # Iterate over each image from the sample
  for row, image_filename in enumerate(list_of_sample_images):
    # Get image name
    image_name = image_filename.split('.')[0]

    # Get HR image path
    HR_image_path = os.path.join(image_folder_HR, image_filename)

    # Get LR image path
    LR_image_filename = f"{image_name}_x4.{image_filename.split('.')[-1]}"    #Using the same file extension as the HR counterpart
    LR_image_path = os.path.join(image_folder_LR, LR_image_filename)

    # Loading LR and HR images
    LR_image = Image.open(LR_image_path)        #Alternatively:   LR = mpimg.imread(LR_image_path)    where:  import matplotlib.image as mpimg
    HR_image = mpimg.imread(HR_image_path)
    HR_array = cv2.imread(HR_image_path)

    # Plot the original image for the current row
    axs[row, 0].imshow(HR_image)
    axs[row, 0].axis('off')
    axs[row, 0].set_title(f"Original\n{image_name}\n", fontsize=fontsize)

    # Plot the bicubic upscaled image for the LR image of the current row
    bicubic_image = get_bicubic_image(LR_image, scale=scale)
      # Convert to array to measure with respect to the original one
    bicubic_image_array = image_to_array(bicubic_image, temp_folder=save_to)
    MSE = calc_mse(HR_array,bicubic_image_array)
    PSNR = calc_psnr_simplified(bicubic_image_array, HR_array, scale=scale, rgb_range=rgb_range, crop=crop)
    SSIM = calc_ssim(HR_array,bicubic_image_array,channel_axis=2)
      # Plot it
    axs[row, 1].imshow(bicubic_image)
    axs[row, 1].axis('off')
    axs[row, 1].set_title(f"Bicubic\nPSNR {PSNR:.2f} dB\nMSE {MSE:.1f} | SSIM: {SSIM:.2f}", fontsize=fontsize)
    
    # Plot the upscaled image for each different EDSR
    for edsr_idx, pretrain_path in enumerate(list_of_edsr_pretrain_paths):
      if verbose :
        print(f"[INFO] Applying EDSR from {pretrain_path} to image {image_filename} in directory {dir_src}.")
      # Apply EDSR model
      SR_array = apply_EDSR_to_image(LR_image, folder=current_image_folder, data_test="Demo", 
                                     scale=scale, pretrain_path=pretrain_path,
                                     dir_src=dir_src, verbose=False, keep_on_disk=False)
      # Get SR image to plot
      SR_image = array_to_image(SR_array, temp_folder=save_to)
      # Measure with respect to the original
      MSE = calc_mse(HR_array,SR_array)
      PSNR = calc_psnr_simplified(SR_array, HR_array, scale=scale, rgb_range=rgb_range, crop=crop)
      SSIM = calc_ssim(HR_array,SR_array,channel_axis=2)
      # Plot it
      edsr_name = list_of_edsr_names[edsr_idx]
      axs[row, edsr_idx+2].imshow(SR_image)
      axs[row, edsr_idx+2].axis('off')
      axs[row, edsr_idx+2].set_title(f"{edsr_name}\nPSNR {PSNR:.2f} dB\nMSE {MSE:.1f} | SSIM: {SSIM:.2f}", fontsize=fontsize)


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
path_to_save = os.path.join(dir_images, "CompositionFromSamples")         # ./images/CompositionFromSamples
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save, exist_ok=True)
current_image_folder = os.path.join(dir_images, "current_image_display")  # ./images/current_image_display

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
rgb_range = 255
crop = True
verbose = False
verbose_comp = True
start_image = 30972
end_image = 32261
number_of_samples = 5     # Number of images to take for the sample

# Figure parameters
fig_size = (10, 10)
fontsize = 8
show_fig = False




###################
# Get list of image files in the folder
list_of_images_paths = retrieve_filepaths(image_folder_HR, file_extension=file_extension)
n_images = len(list_of_images_paths)

# Restrict the images to the corresponding range
list_of_images_paths_in_range = list_of_images_paths[start_image-1:end_image-1]

# Select a random sample from the range that can be used for comparison (i.e., their dimensions (width, height) are multiple of the scale of the model)
sample_of_images = select_random_images_if_multiple_of(list_of_images_paths_in_range, 
                                                       num_images=number_of_samples, multiple_of=scale)


# Make composition from sample
range_start_formatted = make_numbers_the_same_length(start_image, n_images)
range_end_formatted = make_numbers_the_same_length(end_image, n_images)
comp_name = f"composition_from_sample_in_range{range_start_formatted}-{range_end_formatted}"

make_composition_from_sample(list_of_sample_images=sample_of_images, 
                            image_folder_HR=image_folder_HR, image_folder_LR=image_folder_LR,
                            list_of_edsr_pretrain_paths=list_of_edsr_pretrain_paths, 
                            list_of_edsr_names=list_of_edsr_names,
                            scale=scale, rgb_range=rgb_range, crop=crop,
                            comp_name=comp_name, save_to=path_to_save,
                            current_image_folder=current_image_folder,
                            dir_src=dir_src, verbose=verbose_comp,
                            fig_size=fig_size, fontsize=fontsize, show_fig=show_fig)
if verbose :
    print(f"Composition for range {range_start_formatted}-{range_end_formatted} has been saved as {comp_name} in {path_to_save}.")

