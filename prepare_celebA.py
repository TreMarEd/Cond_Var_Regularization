import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import PIL

# train and vali: n = 20'000 c= 5'000, all Y=1 degraded
# test1 n= 5344, all Y=1 degraded
# test2 n = 5344 all Y=0 degraded
# images with shape (3, 218, 178), why does heinze have 3 x 64 x 48?? 
# try to create your own Celeb dataset with different resolution using PIL https://stackoverflow.com/questions/62282695/reduce-image-resolution
# also automatically degrade its quality using the following command. unclear: willdatasets.CelebA still read it or does it checksum it?
# use raw fstrins: https://stackoverflow.com/questions/58302531/combine-f-string-and-raw-string-literal
result = subprocess.run(["magick", "convert", "-quality", "60", r"C:\Users\Marius\Desktop\DAS\000001.jpg", 
                         r"C:\Users\Marius\Desktop\DAS\000001__.jpg"], capture_output = True, text = True)

#from PIL import Image
# resizing an image with PIL
#https://stackoverflow.com/questions/9174338/programmatically-change-image-resolution
#im = Image.open("test.png")
#im.save("test-600.png", dpi=(600,600))


# attributes and their index for me to use
# Eyeglasses 15
# mustache 22
# Wearing_Hat 34
# Wearing_Earrings 35


CelebA_data = datasets.CelebA(
    root=".\CelebA",
    split='all',
    target_type='attr',
    transform=ToTensor(),
    download=True
)



print("hoi")