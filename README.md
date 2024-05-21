## About this repository
This repository has been made as part of the work **"Implementation of the Enhanced Deep Super-Resolution (EDSR) Algorithm on pathological images for image generation applications"** (2024).

We employed the work of "Enhanced Deep Residual Networks for Single Image Super-Resolution" (CVPRW 2017), which Pytorch Implementation present in the repository [EDSR-Pytorch](https://github.com/sanghyun-son/EDSR-PyTorch), from:

[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](https://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]

For more detailed information about this model, please refer to its original authors.


## Our work
We focused on the single 4-times scale of the EDSR model on pathological images. We trained the EDSR x4 model using 2 different datasets, each composed of High-Resolution (HR) images and their corresponding Low-Resolution (LR) downscalings (by a factor 4), studied the performance in terms of the image quality between the Super-Resolution (SR) upscalings of the LR images compared to the HR versions, and applied our last trained model to synthetic generated pathological images.

In our work we also made some modifications to the original code, added two new files ([freeze.py](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/EDSR-PyTorch/src/freeze.py) and [main_use.py](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/EDSR-PyTorch/src/main_use.py)) and a new argument (save_models_each). These modifications were meant to fit our work, and are in no means improvements or replacements of any kind to the original work made by the authors of the model. We also produced a sort of Documentation of our understanding of the PyTorch Implementation of the EDSR x4 model, as well as the pertinent explanations of our contributions.


## The content of this repository
In this repository we have the following folders and files:

* [EDSR-PyTorch](https://github.com/giancarlocuticchia/Master-sThesis/tree/main/EDSR-PyTorch): Folder with our cloned version of the original [EDSR-Pytorch](https://github.com/sanghyun-son/EDSR-PyTorch) repository, with our modified and added files.
* [Notebooks-scripts](https://github.com/giancarlocuticchia/Master-sThesis/tree/main/Notebooks-scripts): Folder containing Jupyter Notebooks and Python scripts we produced in the realization of our work.
* [Output-files](https://github.com/giancarlocuticchia/Master-sThesis/tree/main/Output-files): Folder containing additional files that could serve as examples.
* [Documentation_EDSRPyTorch.pdf](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Documentation_EDSRPyTorch.pdf): PDF file containing the Documentation we produced (corresponding to one of the Appendices of our work).

More information regarding each folder can be found in them.


## Our datasets
To train the model we used two different dataset:

* <u>General Dataset (TCIA)</u>: 2500 patches (in PNG format) of pathological images from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/) (TCIA). These dataset is [available in Kaggle](https://www.kaggle.com/datasets/giancarlocuticchia/general-dataset-tcia) (22.66 GB).
* <u>Dedicated Dataset (Humanitas)</u>: 32261 patches (in PNG format) of pathological images by the Humanitas Research Institute (Milan, Italy).

More information about how these datasets were prepared can be seen in our [dedicated Jupyter Notebook](https://github.com/giancarlocuticchia/Master-sThesis/blob/main/Notebooks-scripts/Notebooks/1_Preparing_the_datasets.ipynb).


## Trained models
We performed training on the EDSR x4 model by taking as starting point the pre-trained EDSR x4 model by its authors. We first trained it in our General Dataset (TCIA) and then with trained further in our Dedicated Dataset (Humanitas) in two ways: one providing the whole dataset in a single run, and one providing the dataset in batches (sequentially). The final trained models for each case are:

* <u>edsr_x4-4f62e9ef.pt</u>: Pre-trained EDSR x4 by its authors. It can be obtained from their [original source](https://cv.snu.ac.kr/research/EDSR/models/edsr_x4-4f62e9ef.pt) and from [our source in Kaggle](https://www.kaggle.com/datasets/giancarlocuticchia/pretrained-edsr-x4-models?select=edsr_x4-4f62e9ef.pt) (172.38 MB)
* <u>edsr_x4-best_trained_general.pt</u>: EDSR x4 trained in General Dataset (TCIA). Available from [our source in Kaggle](https://www.kaggle.com/datasets/giancarlocuticchia/pretrained-edsr-x4-models?select=edsr_x4-best_trained_general.pt) (172.41 MB)
* <u>edsr_x4-best_trained_dedicated_whole.pt</u>: EDSR x4 trained in Dedicated Dataset (Humanitas), with the whole dataset in a single run. Available from [our source in Kaggle](https://www.kaggle.com/datasets/giancarlocuticchia/pretrained-edsr-x4-models?select=edsr_x4-best_trained_dedicated_whole.pt) (172.41 MB)
* <u>edsr_x4-best_trained_dedicated_inbatches.pt</u>: EDSR x4 trained in Dedicated Dataset (Humanitas), with the dataset in batches. Available from [our source in Kaggle](https://www.kaggle.com/datasets/giancarlocuticchia/pretrained-edsr-x4-models?select=edsr_x4-best_trained_dedicated_inbatches.pt) (172.41 MB)



