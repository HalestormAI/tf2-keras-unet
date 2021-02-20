# UNet Segmentation Network in TF Keras

This implementation is mostly based on the original paper [U-Net: Convolutional Networks for Biomedical Image Segmentation.](https://arxiv.org/abs/1505.04597) [1].

Trains a segmentation model on the [Oxford IIIT Pets dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) [2].

Uses TF datasets to load the dataset and performs some minimal preprocessing on it (TODO: data augmentation, random crops).

![UNet framework diagram from original paper](readme/unet-diagram.png?raw=true "Title")

Also useful for implementation details were:
  * [Image segmentation with a U-Net-like architecture
](https://keras.io/examples/vision/oxford_pets_image_segmentation/)
  * [Implementation of deep learning framework -- Unet, using Keras
](https://github.com/zhixuhao/unet)
  * [Keras U-Net Starter](https://www.kaggle.com/keegil/keras-u-net-starter-lb-0-277_)

[1] Ronneberger O., Fischer P., Brox T. (2015) U-Net: Convolutional Networks for Biomedical Image Segmentation. In: Navab N., Hornegger J., Wells W., Frangi A. (eds) Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015. MICCAI 2015. Lecture Notes in Computer Science, vol 9351. Springer, Cham.
[2] O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. (2012) Cats and Dogs. IEEE Conference on Computer Vision and Pattern Recognition

