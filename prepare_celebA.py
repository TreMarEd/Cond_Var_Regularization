"""
The following script prepares the CelebA data for the transfer learning experiment using conditional variance regularization for 
domain shift invariance. The data is prepared the following way:
    - the original CelebA data is resized and an additional resized version with randomly degraded images is created

    - train, vali and test sets for the regularized training are created, f.e. for detecting eyeglasses. 
      The goal is to transfer the learned features to the non-regulairzed training
        - in train and vali all Y=0 datapoints are non-degraded and all Y=1 datapoints are degraded. Additionally,
          some Y=1 datapoints are augmented with non-degraded counterparts to learn the domain shif invariance
        - the test1 set contains only non-degraded Y=0 data and only degraded Y=1 data
        - the test2 set contains only degraded Y=0 data and only non-degraded Y=1 data to test domain shift robustness

    - train, vali and test sets are created for the non-regularized training, f.e. for detecting mustaches
        - in train and vali all Y=0 datapoints are non-degraded and all Y=1 datapoints are degraded. 
        - the test1 set contains only non-degraded Y=0 data and only degraded Y=1 data
        - the test2 set contains only degraded Y=0 data and only non-degraded Y=1 data to test domain shift robustness
"""

from torchvision import datasets
from torchvision.transforms import ToTensor
from flax import linen as nn
import numpy as np
import os
import shutil
import subprocess
from PIL import Image
import jax
import jax.numpy as jnp
import warnings
import logging
import train_utils as tu


logging.basicConfig(level=logging.INFO, filename=".\logfile.txt", filemode="w+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")

class CNN_celeba(nn.Module):
    """A simple CNN model."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=2)(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=2)(x)
        x = nn.activation.leaky_relu(x)
        #x = nn.Conv(features=128, kernel_size=(5, 5), strides=2)(x)
        #x = nn.activation.leaky_relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 1)) # shape 2x2x32
        x = x.reshape((x.shape[0], -1))
        # extract the learned representation and return it separately. This is needed for CVR regularization
        r = x
        x = nn.Dense(features=2)(x)
        return x, r



def resize_degrade_CelebA(CelebA_path, resize_0, resize_1, seed):
    '''
    Given the original CelebA dataset and an RNG seed saves a resized version of the original dataset 
    in the same format as the original one and also another resized version where images are randomly degraded 
    according to a normal distibution with mean 30 and variance 100 using ImageMagick.

    Parameters:
        CelebA_path (string): path to a directory containing the original Celeb A dataset in the directory named celeba
        resize_0 (int): new resolution along axis 0. Note that in CelebA axis0 smaller than axis1
        resize_1 (int): new resolution along axis 1. Note that in CelebA axis0 smaller than axis1
        seed (int): seed for jax rng creation when degrading images randomly

    Returns:
        None, saves the resized and resized + degraded datasets in the same format and the same directory 
        as CelebA_path is saved in
    '''

    print(f"########################### CREATING RESIZED AND DEGRADED CELEBA ###########################")

    if not os.path.exists(CelebA_path):
        raise Exception("The provided path to the original Celeb A dataset does not exist.")

    # create the relevant directories and copy all relevant files
    orig_path = CelebA_path + r"\celeba"
    resized_path = CelebA_path + fr"_resized{resize_0}x{resize_1}_seed{seed}\celeba"
    resized_degraded_path = CelebA_path + fr"_resized{resize_0}x{resize_1}_degraded_seed{seed}\celeba"

    if not os.path.exists(resized_path + r"\img_align_celeba"):
        os.makedirs(resized_path + r"\img_align_celeba")

    if not os.path.exists(resized_degraded_path + r"\img_align_celeba"):
        os.makedirs(resized_degraded_path + r"\img_align_celeba")

    # only txt files should be copied, and not the original images
    txt_files = [f for f in os.listdir(orig_path) if os.path.isfile(os.path.join(orig_path, f))]

    for f in txt_files:
        shutil.copyfile(os.path.join(orig_path, f), os.path.join(resized_path, f))
        shutil.copyfile(os.path.join(orig_path, f), os.path.join(resized_degraded_path, f))

    # total number of CelebA images
    n_tot = 202599

    for i in range(1, n_tot+1):
        print(i)
        i = str(i)
        # string with the right number of zeros for the celeb image name
        zs = (6 - len(i)) * "0"
        image_name = zs + i + ".jpg"

        orig_im_path = orig_path + fr"\img_align_celeba\{image_name}"
        resized_im_path = resized_path + fr"\img_align_celeba\{image_name}"
        deg_im_path = resized_degraded_path + fr"\img_align_celeba\{image_name}"

        im = Image.open(orig_im_path)
        im_resized = im.resize((resize_0, resize_1))
        im_resized.save(resized_im_path)

        quality = -100
        # quality should be a positive number
        key = jax.random.key(seed)
        while quality < 0:
            key, subkey = jax.random.split(key)
            # degrade with normal distribution with mean 30 and Variance 10^2
            quality = 30 + 10 * jax.random.normal(subkey, (1,))

        quality = str(int(quality[0]))
        # run the Image Magick command for image degradatin using normally distributed quality
        subprocess.run(["magick", "convert", "-quality", quality, resized_im_path, deg_im_path],
                       capture_output=True, text=True)

    return None


def sample_arrays(arrays, n, key, axis=0):
    '''
    Given a list of arrays, samples values from the same random n indices of all of them and returns a list of arrays both
    with the sampled data and a list of arrays with containing the remaining non-sampled data

    Parameters:
        base_path (list): list of arrays to be sampled
        n (int): number of samples to draw
        key (jax RNG key): rng key to be used by jax
        axis (int): axis along which the samples will be drawn

    Returns:
        sample_arrays (list): list of arrays containing the samlpled entries
        rest_arrays (list): list of arrays containing the remaining non-sampled data
    '''

    shapes = [jnp.shape(array) for array in arrays]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All input arrays need to have the same shape")

    idxs = jnp.arange(shapes[0][axis])
    key, subkey = jax.random.split(key)
    sample = jax.random.choice(subkey, idxs, shape=(n,), replace=False)

    sample_arrays = [jnp.take(array, sample, axis=axis) for array in arrays]
    rest_arrays = [jnp.delete(array, sample, axis=axis) for array in arrays]

    return sample_arrays, rest_arrays


def create_augmented_CelebA(base_path, n_train, n_vali, n_test, f_1, f_aug, label_idx, resize_0, resize_1, seed):
    '''
    Provided paths to datasets of degraded and non-degraded CelebA images, creates and persists the augmentd Celeb A 
    dataset for the conditional variance regularization experiment. The created dataset contains a test and validation 
    set containing:
        - A total of n original, non-augmented datapoint, whereof f_1 * n will have Y=1, and (1-f_1) * n will have Y=0
        - all datapoints with Y=0 will be non-degraded, whereas (1-f_aug) * f_1 * n datapoints with Y=1 will be degraded
        - additionally, the data set will contain f_aug * f_1*n pairs of datapoints representing the same image with one 
          datapoint being degraded and the other being non-degraded. These pairs will be referred to as dublettes, the 
          non-paired data as singletts.
    Also a test1 and test2 data set will be created where in test1 all Y=0 datapoints will be non-degraded, whereas Y=1 will
    degraded. Test2 is the opposite: all Y=0 are degraded, all Y=1 non-degraded

    For train and vali data, the singlett and dublett data is saved separately:
        - x_<train/vali>_sing: features of non-augmented datapoints
        - y_<train_vali>_sing: labels of non-augmented datapoints
        - x_<train/vali>_orig: the features of all original, meaning non-augmented, degraded datapoints
        - x_<train/vali>_aug: the features of all augmented, meaning non-degraded datapoints
        - y_<train/vali>_orig: the labels of all original, meaning non-augmented, degraded datapoints

    Parameters:
        base_path (string): path to a directory containing the CelebA directories created by the function resize_degrade_CelebA
        n_train (int): the number of non-augmented datapoints in the train set
        n_vali (int): the number of non-augmented datapoints in the vali set
        n_test (int): the number of datapoints in the test1 and test2 sets
        f_1 (float): n*f_1 is the number of non-augmented datapoints with Y=1
        f_aug (float): fraction of Y=1 datapoints that are to be aumented with non-degraded images in the train and vali sets
        label_idx (int): the index of the relevant label to be used from the original CelebA dataset, f.e. 15 => eyeglasses
        seed (int): seed that was used during the call to resize_degrade_CelebA to create the prepared CelebA datasets

    Returns:
        None

    '''
    
    print(f"########################### CREATING AUGMENTED CELEBA FOR LABEL {label_idx} ###########################")

    n_tot = 202599
    assert n_train + n_vali + 2*n_test < n_tot, "train, test and vali set have bigger combined size than Celeb A"

    if (n_train + n_vali + 2*n_test) * f_1 < 0.04 * n_tot:
        warnings.warn("It is likely that you requested more Y=1 data than is contained in Celeb A. In this \
                      case a downstream error in the sample_arrays function will be raised")

    key = jax.random.key(seed)
    CelebA = datasets.CelebA(root=base_path + f".\CelebA_resized{resize_0}x{resize_1}_seed{seed}", split='all', target_type='attr',
                             transform=ToTensor(), download=True)
    CelebA_d = datasets.CelebA(root=base_path + f".\CelebA_resized{resize_0}x{resize_1}_degraded_seed{seed}", split='all', target_type='attr',
                               transform=ToTensor(), download=True)
    
    # separate features according to whether Y=0 or Y=1
    # d for degraded
    x_0_d = []
    x_1_d = []
    #nd for non-degraded
    x_0_nd = []
    x_1_nd = []

    for i in range(n_tot):
        print(i)
        
        y = int(CelebA[i][1][label_idx])

        img_d = CelebA_d[i][0].numpy()
        img_nd = CelebA[i][0].numpy()
        # surprisingly, the magick command for image degradation seems to delete some colour channels if not needed anymore
        # after degradation, which results in heterogeneous shape. This is rarely the case (less than 1 in 3000) and I
        # just skip these datapoints here
        if np.shape(img_d)[0] == 3 and np.shape(img_nd)[0] == 3:
            
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

    # number of original Y=1 datapoints in final dataset
    m_train = int(f_1 * n_train)
    m_vali  = int(f_1 * n_vali)
    m_test = int(f_1 * n_test)

    # c is the number of augmented datapoints
    c_train = int(m_train * f_aug)
    c_vali = int(m_vali * f_aug)

    ################################### create train data ###################################
    key, subkey = jax.random.split(key)
    [x_0_d_sample, x_0_nd_sample], [x_0_d, x_0_nd] = sample_arrays([x_0_d, x_0_nd], n_train - m_train, subkey)
    
    key, subkey = jax.random.split(key)
    [x_1_d_sample, x_1_nd_sample], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], m_train - c_train, subkey)
    
    x_train_sing = jnp.vstack((x_0_nd_sample, x_1_d_sample))
    y_train_sing = jnp.hstack((jnp.zeros((n_train-m_train)), jnp.ones((m_train - c_train))))
    y_train_sing = y_train_sing

    key, subkey = jax.random.split(key)
    [x_train_orig, x_train_aug], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], c_train, subkey)
    
    # all dublettes have Y=1
    y_train_orig = jnp.ones((c_train,))

    ################################### create vali data ###################################
    key, subkey = jax.random.split(key)
    [x_0_d_sample, x_0_nd_sample], [x_0_d, x_0_nd] = sample_arrays([x_0_d, x_0_nd], n_vali - m_vali, subkey)
    
    key, subkey = jax.random.split(key)
    [x_1_d_sample, x_1_nd_sample], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], m_vali - c_vali, subkey)
    
    x_vali_sing = jnp.vstack((x_0_nd_sample, x_1_d_sample))
    y_vali_sing = jnp.hstack((jnp.zeros((n_vali-m_vali)), jnp.ones((m_vali - c_vali))))

    key, subkey = jax.random.split(key)
    [x_vali_orig, x_vali_aug], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], c_vali, subkey)
    
    # all dublettes have Y=1
    y_vali_orig = jnp.ones((c_vali,))

    # the vali data does not have to be saved separately: this is only done for the test set to efficiently
    # create training batches during training: the dublettes should be at the end of the batch with data 
    # points in the same group consecutive to each other. One can directly prepare the vali data in this fashion here
    y_vali = jnp.hstack((y_vali_sing, y_vali_orig, y_vali_orig)).astype(jnp.int32)

    # number of singlett data points
    n_s = n_vali - c_vali

    # TODO: find out why you here need to change order of resize_0 and resize_1, apparanetly resize_create_CelebA saves
    # the images with shape (resize_1, resize_0)
    x_vali = jnp.zeros((n_vali + c_vali, resize_1, resize_0, 3))
    x_vali = x_vali.at[:n_s, :, :, :].set(x_vali_sing)

    for j in range(c_vali):
        # first add the original data point
        x_vali = x_vali.at[n_s + 2*j, :, :, :].set(x_vali_orig[j, :, :, :])
        # then add the augmented data point directly afterward
        x_vali = x_vali.at[n_s + (2*j)+1, :, :, :].set(x_vali_aug[j, :, :, :])

    ################################### create test1 and test2 data ###################################
    key, subkey = jax.random.split(key)
    [x_0_d_sample, x_0_nd_sample], [x_0_d, x_0_nd] = sample_arrays([x_0_d, x_0_nd], n_test - m_test, subkey)
    
    key, subkey = jax.random.split(key)
    [x_1_d_sample, x_1_nd_sample], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], m_test, subkey)
    
    x_test1 = jnp.vstack((x_0_nd_sample, x_1_d_sample))
    y_test1 = jnp.hstack((jnp.zeros((n_test - m_test)), jnp.ones((m_test,))))
    y_test1 = y_test1.astype(jnp.int32)

    key, subkey = jax.random.split(key)
    [x_0_d_sample, x_0_nd_sample], [x_0_d, x_0_nd] = sample_arrays([x_0_d, x_0_nd], n_test - m_test, subkey)
    
    key, subkey = jax.random.split(key)
    [x_1_d_sample, x_1_nd_sample], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], m_test, subkey)
    
    x_test2 = jnp.vstack((x_0_d_sample, x_1_nd_sample))
    y_test2 = jnp.hstack((jnp.zeros((n_test - m_test)), jnp.ones((m_test,))))
    y_test2 = y_test2.astype(jnp.int32)
    
    ################################### persist everything ###################################
    dir_path = base_path + fr"\augmented_CelebA_resized{resize_0}x{resize_1}_seed{seed}_label{label_idx}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path + r"\train")
        os.makedirs(dir_path + r"\vali")
        os.makedirs(dir_path + r"\test")

    jnp.save(dir_path + r"\train\x_train_sing.npy", x_train_sing)
    jnp.save(dir_path + r"\train\y_train_sing.npy", y_train_sing)
    jnp.save(dir_path + r"\train\x_train_orig.npy", x_train_orig)
    jnp.save(dir_path + r"\train\x_train_aug.npy", x_train_aug)
    jnp.save(dir_path + r"\train\y_train_orig.npy", y_train_orig)

    jnp.save(dir_path + r"\vali\x_vali.npy", x_vali)
    jnp.save(dir_path + r"\vali\y_vali.npy", y_vali)

    jnp.save(dir_path + r"\test\x_test1.npy", x_test1)
    jnp.save(dir_path + r"\test\y_test1.npy", y_test1)

    jnp.save(dir_path + r"\test\x_test2.npy", x_test2)
    jnp.save(dir_path + r"\test\y_test2.npy", y_test2)

    return None


def create_CelebA(base_path, n_train, n_vali, n_test, f_1, label_idx, resize_0, resize_1, seed):
    '''
    Provided paths to datasets of degraded and non-degraded CelebA images, creates and persists the non-augmentd Celeb A 
    dataset for the conditional variance regularization experiment. 
        - train, vali, test1: contain a total of n datapoints of which f_1 * n have Y=1. All Y=0 datapoints are non-degraded
          and all Y=1 datapoints are degraded
        - test2: same as above, but All Y=1 datapoints are non-degraded and all Y=0 datapoints are degraded

    The data is persisted in the following way:
        - x_<train/vali>: features of non-augmented datapoints
        - y_<train_vali>: labels of non-augmented datapoints

    Parameters:
        base_path (string): path to a directory containing the CelebA directories created by the function resize_degrade_CelebA
        n_train (int): the number of datapoints in the train set
        n_vali (int): the number of datapoints in the vali set
        n_test (int): the number of datapoints in the test1 and test2 sets
        f_1 (float): n*f_1 is the number of datapoints with Y=1
        label_idx (int): the index of the relevant label to be used from the original CelebA dataset, f.e. 15 => eyeglasses
        seed (int): seed that was used during the call to resize_degrade_CelebA to create the prepared CelebA datasets

    Returns:
        None
    '''
    
    print(f"########################### CREATING NON-AUGMENTED CELEBA FOR LABEL {label_idx} ###########################")

    n_tot = 202599
    assert n_train + n_vali + 2*n_test < n_tot, "train, test and vali set have bigger combined size than Celeb A"
    if (n_train + n_vali + 2*n_test) * f_1 < 0.04 * n_tot:
        warnings.warn("It is likely that you requested more Y=1 data than is contained in Celeb A. In this \
                      case a downstream error in the sample_arrays function will be raised")

    key = jax.random.key(seed)
    CelebA = datasets.CelebA(root=base_path + f".\CelebA_resized{resize_0}x{resize_1}_seed{seed}", split='all', target_type='attr',
                             transform=ToTensor(), download=True)
    CelebA_d = datasets.CelebA(root=base_path + f".\CelebA_resized{resize_0}x{resize_1}_degraded_seed{seed}", split='all', target_type='attr',
                               transform=ToTensor(), download=True)
    
    # separate features according to whether Y=0 or Y=1
    # d for degraded
    x_0_d = []
    x_1_d = []
    #nd for non-degraded
    x_0_nd = []
    x_1_nd = []

    for i in range(n_tot):
        print(i)
        
        y = int(CelebA[i][1][label_idx])

        img_d = CelebA_d[i][0].numpy()
        img_nd = CelebA[i][0].numpy()
        # surprisingly, the magick command for image degradation seems to delete some colour channels if not needed anymore
        # after degradation, which results in heterogeneous shape. This is rarely the case (less than 1 in 3000) and I
        # just skip these datapoints here
        if np.shape(img_d)[0] == 3 and np.shape(img_nd)[0] == 3:
            
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

    # number of original Y=1 datapoints in final dataset
    m_train = int(f_1 * n_train)
    m_vali  = int(f_1 * n_vali)
    m_test = int(f_1 * n_test)

    ################################### create train data ###################################
    key, subkey = jax.random.split(key)
    [x_0_d_sample, x_0_nd_sample], [x_0_d, x_0_nd] = sample_arrays([x_0_d, x_0_nd], n_train - m_train, subkey)
    
    key, subkey = jax.random.split(key)
    [x_1_d_sample, x_1_nd_sample], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], m_train, subkey)
    
    x_train = jnp.vstack((x_0_nd_sample, x_1_d_sample))
    y_train = jnp.hstack((jnp.zeros((n_train - m_train)), jnp.ones((m_train,))))
    y_train = y_train.astype(jnp.int32)
    
    ################################### create vali data ###################################
    key, subkey = jax.random.split(key)
    [x_0_d_sample, x_0_nd_sample], [x_0_d, x_0_nd] = sample_arrays([x_0_d, x_0_nd], n_vali - m_vali, subkey)
    
    key, subkey = jax.random.split(key)
    [x_1_d_sample, x_1_nd_sample], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], m_vali, subkey)
    
    x_vali = jnp.vstack((x_0_nd_sample, x_1_d_sample))
    y_vali = jnp.hstack((jnp.zeros((n_vali - m_vali)), jnp.ones((m_vali,))))
    y_vali = y_vali.astype(jnp.int32)

    ################################### create test1 and test2 data ###################################
    key, subkey = jax.random.split(key)
    [x_0_d_sample, x_0_nd_sample], [x_0_d, x_0_nd] = sample_arrays([x_0_d, x_0_nd], n_test - m_test, subkey)
    
    key, subkey = jax.random.split(key)
    [x_1_d_sample, x_1_nd_sample], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], m_test, subkey)
    
    x_test1 = jnp.vstack((x_0_nd_sample, x_1_d_sample))
    y_test1 = jnp.hstack((jnp.zeros((n_test - m_test)), jnp.ones((m_test,))))
    y_test1 = y_test1.astype(jnp.int32)

    key, subkey = jax.random.split(key)
    [x_0_d_sample, x_0_nd_sample], [x_0_d, x_0_nd] = sample_arrays([x_0_d, x_0_nd], n_test - m_test, subkey)
    
    key, subkey = jax.random.split(key)
    [x_1_d_sample, x_1_nd_sample], [x_1_d, x_1_nd] = sample_arrays([x_1_d, x_1_nd], m_test, subkey)
    
    x_test2 = jnp.vstack((x_0_d_sample, x_1_nd_sample))
    y_test2 = jnp.hstack((jnp.zeros((n_test - m_test)), jnp.ones((m_test,))))
    y_test2 = y_test2.astype(jnp.int32)
    
    ################################### persist everything ###################################
    dir_path = base_path + fr"\nonaugmented_CelebA_resized{resize_0}x{resize_1}_seed{seed}_label{label_idx}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path + r"\train")
        os.makedirs(dir_path + r"\vali")
        os.makedirs(dir_path + r"\test1")
        os.makedirs(dir_path + r"\test2")

    jnp.save(dir_path + r"\train\x_train.npy", x_train)
    jnp.save(dir_path + r"\train\y_train.npy", y_train)

    jnp.save(dir_path + r"\vali\x_vali.npy", x_vali)
    jnp.save(dir_path + r"\vali\y_vali.npy", y_vali)

    jnp.save(dir_path + r"\test\x_test1.npy", x_test1)
    jnp.save(dir_path + r"\test\y_test1.npy", y_test1)

    jnp.save(dir_path + r"\test\x_test2.npy", x_test2)
    jnp.save(dir_path + r"\test\y_test2.npy", y_test2)

    return None


def load_celeba(augmented, base_path, resize_0, resize_1, seed, label_idx):
    '''
    Parameters:
        augmented (bool): boolean stating whether to load an augmented or non-augmented dataset
        base_path (strin): path containing the directory created by a previous call to create_CelebA or
                           create_augmented_CelebA
        resize_0 (int): the resolution along the first axis of the images to be loaded
        resize_1 (int): the resolution along the second axis of the images to be loaded
        seed (int): seed that was used to create the dataset to be loaded in a previous call to create_CelebA or
                    create_augmented_CelebA
        label_idx (int): index of the label that was used to create the dataset, f.e. 15 = eyeglasses
       
    Returns:
        train_data (dic): dictionary with keys "sing_features", "sing_labels", "dub_orig_featrues", "dub_aug_features", "dub_labels".
    '''
    if augmented:
        dir_path = base_path + fr"\augmented_CelebA_resized{resize_0}x{resize_1}_seed{seed}_label{label_idx}"
    else: 
        dir_path = base_path + fr"\nonaugmented_CelebA_resized{resize_0}x{resize_1}_seed{seed}_label{label_idx}"

    if not os.path.exists(dir_path):
        raise OSError(2, 'No such file or directory', dir_path)

    print("\n#################### LOADING AUGMENTED CELEBA DATA #################### \n")
    if augmented:
        x_train_sing = jnp.load(dir_path + "\\train\\x_train_sing.npy")
        y_train_sing = jnp.load(dir_path + "\\train\\y_train_sing.npy")
        x_train_orig = jnp.load(dir_path + "\\train\\x_train_orig.npy")
        y_train_orig = jnp.load(dir_path + "\\train\\y_train_orig.npy")
        x_train_aug = jnp.load(dir_path + "\\train\\x_train_aug.npy")
    
    else:
        x_train = jnp.load(dir_path + "\\train\\x_train.npy")
        y_train = jnp.load(dir_path + "\\train\\y_train.npy")

    x_vali = jnp.load(dir_path + "\\vali\\x_vali.npy")
    y_vali = jnp.load(dir_path + "\\vali\\y_vali.npy")

    x_test1 = jnp.load(dir_path + "\\test\\x_test1.npy")
    x_test2 = jnp.load(dir_path + "\\test\\x_test2.npy")
    y_test1 = jnp.load(dir_path + "\\test\\y_test1.npy")
    y_test2 = jnp.load(dir_path + "\\test\\y_test2.npy")

    if augmented:
        train_data = {"sing_features": x_train_sing, "sing_labels": y_train_sing, "dub_orig_features": x_train_orig,
                      "dub_labels": y_train_orig, "dub_aug_features": x_train_aug}
        
    else:
        train_data = {"features": x_train, "labels": y_train}

    vali_data = {"features": x_vali, "labels": y_vali}

    test1_data = {"features": x_test1, "labels": y_test1}
    test2_data = {"features": x_test2, "labels": y_test2}

    return train_data, vali_data, test1_data, test2_data


if __name__ == "__main__":

    # if not already done, download original dataset using
    # datasets.CelebA(root=f".\CelebA", split='all', target_type='attr', transform=ToTensor(), download=True)

    resize_0 = 48
    resize_1 = 64
    seed = 5297
    base_path = r"C:\Users\Marius\Desktop\DAS\Cond_Var_Regularization"
    n_train = 20000
    #n_train = 1000
    n_vali = 5000
    #n_vali = 1000
    n_test = 5000
    #n_test = 1000
    f_1 = 0.25
    f_aug = 0.08
    # attributes and their index for me to use and the count statistics in the dataset:
    # Eyeglasses: (15, 13193), mustache: (22, 8417), Wearing_Hat: (35, 9818)   
    aug_label = 15 #eyeglasses 
    non_aug_label = 22 # mustaches
    CelebA_path = base_path + r"\CelebA"
    

    ################## DEFINE FREE PARAMETES  ##################
    num_epochs = 20
    learning_rate = 0.004

    # n is the number of original data points in training set. Needs to be an integer multiple of 100 
    # for below parameter choices to be optimal: batches are sorted such that always the last 2*d datapoints
    # are augmented groups, meaning d consecutive pairs with first the original data point, 
    # then the augmented/rotated datapoint. The below choice of batch size, d and number of batches ensures that
    # the final batch is not partially filled

    # number of data points to be augmented by rotation, equal to number of dublette groups in the final data set
    #c = 2600
    #c = 130
    batch_size = 204
    #batch_size = 113
    # d is the number of dublette (Y, ID) groups per batch
    d = 4
    #d = 13
    num_batches = 100
    #num_batches  = 10

    # regularization parameters on which to perform model selection
    ls = [0.001, 0.1, 1]

    #resize_degrade_CelebA(CelebA_path, resize_0, resize_1, seed)
    #create_augmented_CelebA(base_path, n_train, n_vali, n_test, f_1, f_aug, aug_label, resize_0, resize_1, seed)

    # need to reduce vali and test size as there are not enough mustaches in the dataset
    #n_vali = 5000
    #n_test = 3461
    #create_CelebA(base_path, n_train, n_vali, n_test, f_1, non_aug_label, resize_0, resize_1, seed)

    train_data, vali_data, test1_data, test2_data = load_celeba(True, base_path, resize_0, resize_1, seed, aug_label)
    #train_data_, vali_data_, test1_data_, test2_data_ = load_celeba(False, base_path, resize_0, resize_1, seed, non_aug_label)
    
    ################## TRAIN MODELS ##################
    cnn = CNN_celeba()
    
    # run unregularized case as model selection with only l=0 to choose from, method chosen does not matter for l=0
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    
    state, t1_accuracy, t2_accuracy = tu.model_selection(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, 
                                                      learning_rate, batch_size, num_batches, 100, d, [0], subkey,
                                                      size_0=64, size_1=48, ccs=3, method="CVP", tf_seed=0)
    
    # select regularization parameter for conditional variance of prediction
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    
    
    state_cvp, t1_accuracy_cvp, t2_accuracy_cvp = tu.model_selection(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, 
                                                                  learning_rate, batch_size, num_batches, 100, d, ls, subkey, 
                                                                  size_0=64, size_1=48, ccs=3, method="CVP", tf_seed=0)
    
    # select regularization parameter for conditional variance of representation
    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    state_cvr, t1_accuracy_cvr, t2_accuracy_cvr = tu.model_selection(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, 
                                                                  learning_rate, batch_size, num_batches, 100, d, ls, subkey, 
                                                                  size_0=64, size_1=48, ccs=3, method="CVR", tf_seed=0)
    
    logging.info("\n###########################################################################\n")
    logging.info(f"NON-REGULARIZED NON-SHIFTED TEST ACCURACY = {t1_accuracy}")
    logging.info(f"CVP NON-SHIFTED TEST ACCURACY = {t1_accuracy_cvp}")
    logging.info(f"CVR NON-SHIFTED TEST ACCURACY = {t1_accuracy_cvr}")

    logging.info(f"\nNON-REGULARIZED SHIFTED TEST ACCURACY = {t2_accuracy}")
    logging.info(f"CVP SHIFTED TEST ACCURACY = {t2_accuracy_cvp}")
    logging.info(f"CVR SHIFTED TEST ACCURACY = {t2_accuracy_cvr}")
    
    print("\n###########################################################################\n")
    print(f"NON-REGULARIZED NON-SHIFTED TEST ACCURACY = {t1_accuracy}")
    print(f"CVP NON-SHIFTED TEST ACCURACY = {t1_accuracy_cvp}")
    print(f"CVR NON-SHIFTED TEST ACCURACY = {t1_accuracy_cvr}")

    print(f"\nNON-REGULARIZED SHIFTED TEST ACCURACY = {t2_accuracy}")
    print(f"CVP SHIFTED TEST ACCURACY = {t2_accuracy_cvp}")
    print(f"CVR SHIFTED TEST ACCURACY = {t2_accuracy_cvr}")
    
 