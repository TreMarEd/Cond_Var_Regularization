import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
from PIL import Image








# 000001 to 202599
# train and vali: n = 20'000 c= 5'000, all Y=1 degraded
# test1 n= 5344, all Y=1 degraded
# test2 n = 5344 all Y=0 degraded
# images with shape (3, 218, 178), why does heinze have 3 x 64 x 48?? 
# try to create your own Celeb dataset with different resolution using PIL https://stackoverflow.com/questions/62282695/reduce-image-resolution
# also automatically degrade its quality using the following command. unclear: willdatasets.CelebA still read it or does it checksum it?
# use raw fstrins: https://stackoverflow.com/questions/58302531/combine-f-string-and-raw-string-literal

base_path = r"C:\Users\Marius\Desktop\DAS\Cond_Var_Regularization"
# number of CelebA images
n = 202599

for i in range(1, n+1):
    print(i, "\n")
    i = str(i)
    # string with the right number of zeros for the celeb image name
    zs = (6 - len(i)) * "0"
    image_name = zs + i + ".jpg"

    # TODO: create dir if does not exist

    orig_im_path = base_path + fr"\CelebA\celeba\img_align_celeba\{image_name}"
    resized_im_path = base_path + fr"\CelebA_resized\celeba\img_align_celeba\{image_name}"
    deg_im_path = base_path + fr"\CelebA_resized_degraded\celeba\img_align_celeba\{image_name}"

    im = Image.open(orig_im_path)
    im_resized = im.resize((48, 64))
    im_resized.save(resized_im_path)
    quality = -10
    # quality should be a positive number
    while quality < 0:
        quality = np.random.normal(loc=30., scale=10.0)
    quality = str(int(quality))
    result = subprocess.run(["magick", "convert", "-quality", quality, resized_im_path, deg_im_path], 
                            capture_output = True, text = True)

# TODO: randomly sample 2*20'000 + 2*5344 different indices: 20'000 for train, 20'000 for vali, 5344 for test1 and 5344 for test2

# for train and vali take all Y=0 from non-degraded, and all Y=1 from degraded
# sample c=5'000 Y=1 indices and take their counterparts from the non-degraded dataset
# save the same way as in MNIST: x and  y seperately, augmented and original data separately

# for test1: sample take all Y=0 images from non-degraded, and all Y=1 images from degraded
# for test1: sample take all Y=0 images from degraded, and all Y=0 images from degraded

# persist the datasets created this way and repeat for like 3 different seeds


# attributes and their index for me to use
# Eyeglasses 15
# mustache 22
# Wearing_Hat 34
# Wearing_Earrings 35


# unclear: will datasets.CelebA refuse the resized images? Does it make checksums?
"""
CelebA_data = datasets.CelebA(
    root=".\CelebA",
    split='all',
    target_type='attr',
    transform=ToTensor(),
    download=True
)


print("hoi")
"""