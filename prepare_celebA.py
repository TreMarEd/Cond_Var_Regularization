from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import os
import subprocess
from PIL import Image
import jax
import jax.numpy as jnp

"""
TODO: rewrite the whole thing the following way:
define a function that takes as input:
    - an iterable of indices from which indices are to be sampled
    - a list of arrays on which to evaluate the sampled indices
    - the number of samples to be drawn from the list of indices
    - seed
and returns:
    - a list of indices from which the sampled indices have been deleted
    - a list of arrays corresponding to the input list evaluated at the sampled indices

TODO: write creation of Hat and Mustache dataset

"""

def name():
    pass


base_path = r"C:\Users\Marius\Desktop\DAS\Cond_Var_Regularization"

# number of CelebA images
n_tot = 202599
# resized size of the image along axis 0
resized_0 = 48
# resized size of the image along axis 1
resized_1 = 64

if not os.path.exists(base_path + r"\CelebA_resized") and os.path.exists(base_path + r"\CelebA_resized_degraded"):

    for i in range(1, n_tot + 1):
        
        i = str(i)
        # string with the right number of zeros for the celeb image name
        zs = (6 - len(i)) * "0"
        image_name = zs + i + ".jpg"

        orig_im_path = base_path + fr"\CelebA\celeba\img_align_celeba\{image_name}"
        resized_im_path = base_path + fr"\CelebA_resized\celeba\img_align_celeba\{image_name}"
        deg_im_path = base_path + fr"\CelebA_resized_degraded\celeba\img_align_celeba\{image_name}"

        im = Image.open(orig_im_path)
        im_resized = im.resize((resized_0, resized_1))
        im_resized.save(resized_im_path)

        quality = -10
        # quality should be a positive number
        while quality < 0:
            quality = np.random.normal(loc=30., scale=10.0)

        quality = str(int(quality))
        result = subprocess.run(["magick", "convert", "-quality", quality, resized_im_path, deg_im_path], 
                                capture_output = True, text = True)

CelebA = datasets.CelebA(root=".\CelebA_resized_degraded", split='all', target_type='attr', 
                         transform=ToTensor(), download=True)

CelebA_d = datasets.CelebA(root=".\CelebA_resized_degraded", split='all', target_type='attr', 
                                  transform=ToTensor(), download=True)

################################### CREATE EYEGLASS DATASETS ###################################
"""
for train and vali take all Y=0 from non-degraded, and all Y=1 from degraded.
sample c=5'000 Y=1 indices and take their counterparts from the non-degraded dataset
save the same way as in MNIST: x and  y seperately, augmented and original data separately

for test1: sample take all Y=0 images from non-degraded, and all Y=1 images from degraded
for test1: sample take all Y=0 images from degraded, and all Y=0 images from degraded
attributes and their index for me to use and the count statistics in the dataset: 
Eyeglasses: (15, 13193), mustache: (22, 8417), Wearing_Hat: (35, 9818)
"""

# separate datapoints according to whether Y=0 or Y=1
# for Y=0 use non-degraded data set, for Y=1 degraded dataset. However, for data augmentation non-degraded is also needed for Y=1

# d for degraded
x_0_d = []
x_1_d = []
#nd for non-degraded
x_0_nd = []
x_1_nd = []

for i in range(n_tot):
    print(i)
    # 15th label contains eyeglass info
    y = int(CelebA[i][1][15])

    img_d = CelebA_d[i][0].numpy()
    img_nd = CelebA[i][0].numpy()
    # surprisingly, the magick command for image degradation seems to delete some colour channels if not needed anymore
    # after degradation, which results in heterogeneous shape. This is rarely the case (less than 1 in 3000) and I 
    # just skip these datapoints here
    if np.shape(img_d) == (3, resized_1, resized_0) and np.shape(img_nd) == (3, resized_1, resized_0):

        if y == 1: 
            x_1_d.append(img_d)
            x_1_nd.append(img_nd)

        else:
            x_0_d.append(img_d)
            x_0_nd.append(img_nd)
        
x_0_d = jnp.asarray(x_0_d)
x_0_nd = jnp.asarray(x_0_nd)
x_1_d = jnp.asarray(x_1_d)
x_1_nd = jnp.asarray(x_1_nd)

# color channel dim needs to be the last one to conform with training functions already written
x_0_d = jnp.moveaxis(x_0_d, 1, -1)
x_0_nd = jnp.moveaxis(x_0_nd, 1, -1)
x_1_d = jnp.moveaxis(x_1_d, 1, -1)
x_1_nd = jnp.moveaxis(x_1_nd, 1, -1)

# number of original Y=1 datapoints (meaning wearing glasses).
m_train = 5200
m_vali  = 2600
m_test1 = 2600
m_test2 = 2600

# number of non-augmented datapoints per dataset chosen to be 4*m
alpha = 0.25
n_train = int(m_train/alpha)
n_vali = int(m_vali/alpha)
n_test1 = int(m_test1/alpha)
n_test2 = int(m_test2/alpha)

# fraction of original Y=1 datapoints to be augmented
f = 0.5

# c is the number of augmented datapoints
c_train = int(m_train * f)
c_vali = int(m_vali * f)
# final dataset will have n + m*f datapoints, of which m + m*f will have Y=1, where m*f is the number of augmented datapoints

seed = 3451
key = jax.random.key(seed)
key, subkey = jax.random.split(key)

# total number of Y=0 datapoints that need to be drawn randomly without replacement
num_idxs_0 = (n_train - m_train) + (n_vali - m_vali) + (n_test1 - m_test1) + (n_test2 - m_test2)
idxs_0 = jax.random.choice(subkey, jnp.arange(jnp.shape(x_0_nd)[0]), shape=(num_idxs_0,), replace=False)

idxs_train_sing_0 = idxs_0[:(n_train - m_train)]
idxs_vali_sing_0 = idxs_0[(n_train - m_train):((n_train - m_train) + (n_vali - m_vali))]

idxs_test1_0 = idxs_0[((n_train - m_train) + (n_vali - m_vali)):((n_train - m_train) + (n_vali - m_vali) + (n_test1 - m_test1))]
idxs_test2_0 = idxs_0[-(n_test2 - m_test2):]

key, subkey = jax.random.split(key)
# total number of Y=1 datapoints that need to be drawn randomly without replacement
num_idxs_1 = m_train + m_vali + m_test1 + m_test2
idxs_1 = jax.random.choice(subkey, jnp.arange(jnp.shape(x_1_d)[0]), shape=(num_idxs_1,), replace=False)

idxs_train_sing_1 = idxs_1[:(m_train - c_train)]
idxs_train_dub_1 = idxs_1[(m_train - c_train) : m_train]
idxs_vali_sing_1 = idxs_1[m_train : (m_train + (m_vali - c_vali))]
idxs_vali_dub_1 = idxs_1[(m_train + (m_vali - c_vali)) : (m_train + m_vali)]
idxs_test1_1 = idxs_1[(m_train + m_vali):(m_train + m_vali + m_test1)]
idxs_test2_1 =idxs_1[-(m_test2):]

################################### fill in the data for train ###################################
x_train_sing = jnp.zeros((n_train - c_train, resized_1, resized_0, 3))
y_train_sing = jnp.zeros((n_train - c_train,))

# all dublettes have Y=1
y_train_orig = jnp.ones((c_train,))

# fill the first entries of the singlett data with Y=0 features
tmp = jnp.take(x_0_nd, idxs_train_sing_0, axis=0)
x_train_sing = x_train_sing.at[:(n_train - m_train), :, :, :].set(tmp)
# the corresponding indices of y_train_sing do not need to be adjusted as they are already initialized as 0

# fill the remaining entries of the singlett data with Y=1 features
tmp = jnp.take(x_1_d, idxs_train_sing_1, axis=0)
x_train_sing = x_train_sing.at[(n_train - m_train):, :, :, :].set(tmp)
y_train_sing = y_train_sing.at[(n_train - m_train):].set(jnp.ones(n_train - m_train))

x_train_orig = jnp.take(x_1_d, idxs_train_dub_1, axis=0)
x_train_aug = jnp.take(x_1_nd, idxs_train_dub_1, axis=0)

################################### fill in the data for vali ###################################
x_vali_sing = jnp.zeros((n_vali - c_vali, resized_1, resized_0, 3))
y_vali_sing = jnp.zeros((n_vali - c_vali,))

# all dublettes have Y=1
y_vali_orig = jnp.ones((c_vali,))

# fill the first entries of the singlett data with Y=0 features
tmp = jnp.take(x_0_nd, idxs_vali_sing_0, axis=0)
x_vali_sing = x_vali_sing.at[:(n_vali - m_vali), :, :, :].set(tmp)
# the corresponding indices of y_vali_sing do not need to be adjusted as they are already initialized as 0

# fill the remaining entries of the singlett data with Y=1 features
tmp = jnp.take(x_1_d, idxs_vali_sing_1, axis=0)
x_vali_sing = x_vali_sing.at[(n_vali - m_vali):, :, :, :].set(tmp)
y_vali_sing = y_vali_sing.at[(n_vali - m_vali):].set(jnp.ones(n_vali - m_vali))

x_vali_orig = jnp.take(x_1_d, idxs_vali_dub_1, axis=0)
x_vali_aug = jnp.take(x_1_nd, idxs_vali_dub_1, axis=0)

# initialize all arrays for test1. test sets contain no augmentation scheme, so only a feature and a label array is needed
x_test1 = jnp.zeros((n_test1, resized_1, resized_0, 3))
y_test1 = jnp.zeros((n_test1,))

tmp = jnp.take(x_0_nd, idxs_test1_0, axis=0)
x_test1 = x_test1.at[(n_test1 - m_test1):, :, :, :].set(tmp)
# the corresponding labels in y_test1 are already correctly initialized to 0

tmp = jnp.take(x_1_d, idxs_test1_1, axis=0)
x_test1 = x_test1.at[-m_test1:, :, :, :].set(tmp)
y_test1 = y_test1.at[-m_test1:].set(jnp.ones(m_test1))


# initialize all arrays for test2. test sets contain no augmentation scheme, so only a feature and a label array is needed
x_test2 = jnp.zeros((n_test2, resized_1, resized_0, 3))
y_test2 = jnp.zeros((n_test2,))

tmp = jnp.take(x_0_d, idxs_test1_0, axis=0)
x_test2 = x_test2.at[(n_test2 - m_test2):, :, :, :].set(tmp)
# the corresponding labels in y_test1 are already correctly initialized to 0

tmp = jnp.take(x_1_nd, idxs_test2_1, axis=0)
x_test2 = x_test2.at[-m_test2:, :, :, :].set(tmp)
y_test2 = y_test2.at[-m_test2:].set(jnp.ones(m_test2))

################################### persist everything ###################################

dir_path = base_path + fr"\augmented_CelebA_seed{seed}"
if not os.path.exists(dir_path):
    os.makedirs(dir_path + r"\train")
    os.makedirs(dir_path + r"\vali")
    os.makedirs(dir_path + r"\test1")
    os.makedirs(dir_path + r"\test2")

jnp.save(dir_path + r"\train\x_train_sing.npy", x_train_sing)
jnp.save(dir_path + r"\train\y_train_sing.npy", y_train_sing)
jnp.save(dir_path + r"\train\x_train_orig.npy", x_train_orig)
jnp.save(dir_path + r"\train\x_train_aug.npy", x_train_aug)
jnp.save(dir_path + r"\train\y_train_orig.npy", y_train_orig)

jnp.save(dir_path + r"\vali\x_vali_sing.npy", x_vali_sing)
jnp.save(dir_path + r"\vali\y_vali_sing.npy", y_vali_sing)
jnp.save(dir_path + r"\vali\x_vali_orig.npy", x_vali_orig)
jnp.save(dir_path + r"\vali\x_vali_aug.npy", x_vali_aug)
jnp.save(dir_path + r"\vali\y_vali_orig.npy", y_vali_orig)

jnp.save(dir_path + r"\test1\x_test1.npy", x_test1)
jnp.save(dir_path + r"\test1\y_test1.npy", y_test1)

jnp.save(dir_path + r"\test2\x_test2.npy", x_test2)
jnp.save(dir_path + r"\test2\y_test2.npy", y_test2)











