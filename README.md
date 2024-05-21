## About this repository
This repository has been made as part of the work **"Implementation of the Enhanced Deep Super-Resolution (EDSR) Algorithm on pathological images for image generation applications"** (2024).

We employed the work of "Enhanced Deep Residual Networks for Single Image Super-Resolution" (CVPRW 2017), which Pytorch Implementation present in the repository [EDSR-Pytorch](https://github.com/sanghyun-son/EDSR-PyTorch), from:

[1] Bee Lim, Sanghyun Son, Heewon Kim, Seungjun Nah, and Kyoung Mu Lee, **"Enhanced Deep Residual Networks for Single Image Super-Resolution,"** <i>2nd NTIRE: New Trends in Image Restoration and Enhancement workshop and challenge on image super-resolution in conjunction with **CVPR 2017**. </i> [[PDF](http://openaccess.thecvf.com/content_cvpr_2017_workshops/w12/papers/Lim_Enhanced_Deep_Residual_CVPR_2017_paper.pdf)] [[arXiv](https://arxiv.org/abs/1707.02921)] [[Slide](https://cv.snu.ac.kr/research/EDSR/Presentation_v3(release).pptx)]

For more detailed information about this model, please refer to its original authors.

## Our work
We focused on the single 4-times scale of the EDSR model on pathological images. We trained the EDSR x4 model using 2 different datasets, each composed of High-Resolution (HR) images and their corresponding Low-Resolution (LR) downscalings (by a factor 4), studied the performance in terms of the image quality between the Super-Resolution (SR) upscalings of the LR images compared to the HR versions, and applied our last trained model to synthetic generated pathological images.


