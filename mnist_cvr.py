"""
Author: Marius Tresoldi, Spring 2024
The following script implements conditional variance of representation regularization in jax/flax for domain shift robust 
transfer learning in the MNIST dataset. The general experiment set-up is based on section 5.5 of the following paper:

https://arxiv.org/abs/1710.11469

Training, validation and test set 1 consist of non-rotated digits. Training and validation set additionally contain augmented 
data points that have been rotated, as is required by conditional variance regularization. Test set 2 contains only rotated 
digits. An unregularized CNN is trained, which is not domain shift robust: test1 and test2 accuracies differ
substantially as the network was only trained on non-rotated digits.

Both Conditional variance of prediction (CVP) and conditional variance of representation (CVR) are applied to regularize 
the model and make it domain shift robust and their performance is compared to show that CVR performs only slightly worse than
CVR but with the additional advantage that the representations learned by the model can be used for domain shift robust 
transfer learning, see "celeb_cvr.py".
"""

from flax import linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf
from scipy import ndimage
import os
import numpy as np
import train_utils as tu
import logging

logging.basicConfig(level=logging.INFO, filename=".\logfile_mnist.txt", filemode="w+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


class CNN_mnist(nn.Module):
    """A simple CNN model."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=1)(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=1)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        # extract the learned representation and return it separately. This is needed for CVR regularization
        r = x
        x = nn.Dense(features=10)(x)
        return x, r



def create_aug_mnist(c, seed, n):
    '''
    Given the number of data points to be augmented (c) and an RNG seed returns and saves augmented MNIST data. The data is 
    augmented by randomly selecting <c> data points and rotating them by an angle uniformly distributed in [35, 70] degrees.
    Both training and validation data contain c augmented data points.
    The test1 data consists of the standard domain, where NO image is rotated
    The test2 data consists of the rotated domain, where ALL images are rotated.

    Parameters:
        c (int):    number of datapoints to be augmenteed by rotation
        seed (int): rng seed to be used
        n (int):    number of original data points to use in the training and validation set

    Returns:
        train_data (dic):   dictionary with keys "sing_features", "sing_labels", "dub_orig_featrues", "dub_aug_features", 
                            "dub_labels" containing jax arrays with data. Here "sing" and "dub" refers to singlett and dublette 
                            groups, where a singlett is defined as a (Y, ID) group only containing a single datapoint, 
                            and a dublette a group containing exactly two datapoints, namely the original and the augmented one 
        vali_data (dic):    dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        test1_data (dic):   dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        test2_data (dic):   dictionary with keys "featrues" and "labels", values are jax arrays containing the data

    '''

    print("\n#################### AUGMENTING MNIST DATA #################### \n")

    (x, y), (x_test1, y_test) = tf.keras.datasets.mnist.load_data()

    x = jnp.array(x) / 255
    x_test1 = jnp.array(x_test1) / 255

    # reshape the data to the shapes expected by the training utilities
    x = jnp.reshape(x, (60000, 28, 28, 1))
    x_test1 = jnp.reshape(x_test1, (10000, 28, 28, 1))

    y = jnp.array(y).astype(jnp.int32)
    y_test = jnp.array(y_test).astype(jnp.int32)

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    # sample n indices for the train and n indices for the vali data, 2*n in total
    indices = jax.random.choice(subkey, jnp.arange(60000), shape=(2*n,), replace=False)

    x_train = x[indices[:n], :, :, :]
    y_train = y[indices[:n]]

    x_vali = x[indices[n:], :, :, :]
    y_vali = y[indices[n:]]

    key, subkey = jax.random.split(key)
    # c is the number of datapoints to be augmented by rotation
    aug_indices = jax.random.choice(subkey, jnp.arange(n), shape=(c,), replace=False)

    x_train_orig = x_train[aug_indices, :, :, :]
    y_train_orig = y_train[aug_indices]

    x_train_sing = jnp.delete(x_train, aug_indices, axis=0)
    y_train_sing = jnp.delete(y_train, aug_indices, axis=0)

    x_vali_orig = x_vali[aug_indices, :, :, :]
    y_vali_orig = y_vali[aug_indices]

    x_vali_sing = jnp.delete(x_train, aug_indices, axis=0)
    y_vali_sing = jnp.delete(y_train, aug_indices, axis=0)

    key, subkey = jax.random.split(key)
    # sample c rotation angles for the augmentation
    rot_samples_train = jax.random.uniform(subkey, shape=(c,), minval=35., maxval=70.)
    rot_samples_vali = jax.random.uniform(subkey, shape=(c,), minval=35., maxval=70.)

    x_train_aug = jnp.zeros(jnp.shape(x_train_orig))
    x_vali_aug = jnp.zeros(jnp.shape(x_vali_orig))

    for i in range(c):
        new_img = ndimage.rotate(x_train_orig[i, :, :, :], rot_samples_train[i], reshape=False)
        x_train_aug = x_train_aug.at[i, :, :, :].set(new_img)

        new_img = ndimage.rotate(x_vali_orig[i, :, :, :], rot_samples_vali[i], reshape=False)
        x_vali_aug = x_vali_aug.at[i, :, :, :].set(new_img)

    # two test sets will be used to evaluate domain shift invariance: test set 1 is the original MNIST,
    # test set 2 contains the same images but rotated by 35 or 70 degrees with uniform probability
    key, subkey = jax.random.split(key)
    rot_samples_test = jax.random.uniform(subkey, shape=(10000,), minval=35., maxval=70.)

    x_test2 = x_test1

    for i in range(n):
        new_img = ndimage.rotate(x_test1[i, :, :, :], rot_samples_test[i], reshape=False)
        x_test2 = x_test2.at[i, :, :, :].set(new_img)

    base_path = fr".\data\augmented_mnist\seed{seed}_c{c}_n{n}"

    # prepare vali data s.t. dublette data is at the end with data points in same group consecutive to each other
    n_s = n - c

    x_vali = jnp.zeros((n+c, 28, 28, 1))
    y_vali = jnp.zeros((n+c))

    x_vali = x_vali.at[:n_s, :, :, :].set(x_vali_sing[:n_s, :, :, :])
    y_vali = y_vali.at[:n_s].set(y_vali_sing[:n_s])

    for j in range(c):
        # first add the original data point
        x_vali = x_vali.at[n_s + 2*j, :, :, :].set(x_vali_orig[j, :, :, :])
        # then add the augmented data point directly afterward
        x_vali = x_vali.at[n_s + (2*j)+1, :, :, :].set(x_vali_aug[j, :, :, :])

        y_vali = y_vali.at[n_s + 2*j].set(y_vali_orig[j])
        # then add the augmented data point directly afterward
        y_vali = y_vali.at[n_s + (2*j)+1].set(y_vali_orig[j])

    y_vali = y_vali.astype(jnp.int32)

    if not os.path.exists(base_path):
        os.makedirs(f"{base_path}\\train")
        os.makedirs(f"{base_path}\\vali")
        os.makedirs(f"{base_path}\\test")

    jnp.save(base_path + "\\train\\x_train_sing.npy", x_train_sing)
    jnp.save(base_path + "\\train\\y_train_sing.npy", y_train_sing)
    jnp.save(base_path + "\\train\\x_train_orig.npy", x_train_orig)
    jnp.save(base_path + "\\train\\y_train_orig.npy", y_train_orig)
    jnp.save(base_path + "\\train\\x_train_aug.npy", x_train_aug)

    jnp.save(base_path + "\\vali\\x_vali.npy", x_vali)
    jnp.save(base_path + "\\vali\\y_vali.npy", y_vali)

    jnp.save(base_path + "\\test\\x_test1.npy", x_test1)
    jnp.save(base_path + "\\test\\x_test2.npy", x_test2)
    jnp.save(base_path + "\\test\\y_test.npy", y_test)

    train_data = {"sing_features": x_train_sing, "sing_labels": y_train_sing, "dub_orig_features": x_train_orig,
                  "dub_labels": y_train_orig, "dub_aug_features": x_train_aug}

    vali_data = {"features": x_vali, "labels": y_vali}

    test1_data = {"features": x_test1, "labels": y_test}
    test2_data = {"features": x_test2, "labels": y_test}

    return train_data, vali_data, test1_data, test2_data


def load_aug_mnist(c, seed, n):
    '''
    Given the number of data points to be augmented (c) and an RNG seed loads the augmented MNIST data. 
    If the data has not yet been created, calls "create_aug_mnist". 
    The data is augmented by randomly selecting <c> data points and rotating them by an angle uniformly distributed 
    in [35, 70] degrees. Both training and validation data contain c augmented data points.
    The test1 data consists of the standard domain, where NO image is rotated
    The test2 data consists of the rotated domain, where ALL images are rotated.

    Parameters:
        c (int):    number of datapoints to be augmenteed by rotation
        seed (int): rng seed to be used
        n (int):    number of original data points to use in the training and validation set

    Returns:
        train_data (dic):   dictionary with keys "sing_features", "sing_labels", "dub_orig_featrues", "dub_aug_features", 
                            "dub_labels" containing jax arrays with the data. Here "sing" and "dub" refers to singlett and 
                            dublette groups, where a singlett is defined as a (Y, ID) group only containing a single datapoint, 
                            and a dublette a group containing exactly two datapoints, namely the original and the augmented one
        vali_data (dic):    dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        test1_data (dic):   dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        test2_data (dic):   dictionary with keys "featrues" and "labels", values are jax arrays containing the data
    '''

    base_path = fr".\data\augmented_mnist\seed{seed}_c{c}_n{n}"

    if not os.path.exists(base_path):
        return create_aug_mnist(c, seed, n)

    else:
        print("\n#################### LOADING MNIST DATA #################### \n")
        x_train_sing = jnp.load(base_path + "\\train\\x_train_sing.npy")
        y_train_sing = jnp.load(base_path + "\\train\\y_train_sing.npy")
        x_train_orig = jnp.load(base_path + "\\train\\x_train_orig.npy")
        y_train_orig = jnp.load(base_path + "\\train\\y_train_orig.npy")
        x_train_aug = jnp.load(base_path + "\\train\\x_train_aug.npy")

        x_vali = jnp.load(base_path + "\\vali\\x_vali.npy")
        y_vali = jnp.load(base_path + "\\vali\\y_vali.npy")

        x_test1 = jnp.load(base_path + "\\test\\x_test1.npy")
        x_test2 = jnp.load(base_path + "\\test\\x_test2.npy")
        y_test = jnp.load(base_path + "\\test\\y_test.npy")

        train_data = {"sing_features": x_train_sing, "sing_labels": y_train_sing, "dub_orig_features": x_train_orig,
                      "dub_labels": y_train_orig, "dub_aug_features": x_train_aug}

        vali_data = {"features": x_vali, "labels": y_vali}

        test1_data = {"features": x_test1, "labels": y_test}
        test2_data = {"features": x_test2, "labels": y_test}

    return train_data, vali_data, test1_data, test2_data


if __name__ == "__main__":

    ################## DEFINE FREE PARAMETES  ##################
    # n is the number of original data points in training set. Needs to be integer multiple of 100 
    # for below parameter choices to be optimal: batches are sorted such that always the last 2*d datapoints
    # are augmented groups, meaning d consecutive pairs with first the original data point, 
    # then the augmented/rotated datapoint. The below choice of batch size, d and number of batches ensures that
    # the final batch is not partially filled
    n = 10000
    # fractions of data points to be augmented, should be an integer multiple of 0.01 for below parameter choice to be optimal
    alpha = 0.02
    c = int(alpha * n) # number of data points to be augmented by rotation
    batch_size = int((1 + alpha) * 100)
    d = int(alpha * 100) # the number of dublette (Y, ID) groups per batch
    num_batches = int(n / 100)
    num_epochs = 30
    learning_rate = 0.003
    ls = [0.1, 1, 10, 100] # regularization parameters on which to perform model selection
    seeds = [3229, 6542, 4895, 1008, 5821]

    results = {}
    for key in ["NO-REG", "CVP", "CVR"]:
        results[key] = {"test1": [], "test2": []}

    for seed in seeds:
        ############################## LOAD/CREATE DATA ##############################
        key = jax.random.key(seed)
        train_data, vali_data, test1_data, test2_data = load_aug_mnist(c, seed, n)

        ############################## TRAIN MODELS ##############################
        cnn = CNN_mnist()
        # run unregularized case as model selection with only l=0 to choose from
        state, t1_accuracy, t2_accuracy = tu.model_selection(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, 
                                                            learning_rate, batch_size, num_batches, c, d, [0], key, (28, 28, 1), 
                                                            method="CVP")
        results["NO-REG"]["test1"].append(t1_accuracy)
        results["NO-REG"]["test2"].append(t2_accuracy)

        # select regularization parameter for conditional variance of prediction
        state_cvp, t1_accuracy_cvp, t2_accuracy_cvp = tu.model_selection(cnn, train_data, vali_data, test1_data, test2_data, 
                                                                        num_epochs, learning_rate, batch_size, num_batches, 
                                                                        c, d, ls, key, (28, 28, 1), method="CVP")
        results["CVP"]["test1"].append(t1_accuracy_cvp)
        results["CVP"]["test2"].append(t2_accuracy_cvp)
        
        # select regularization parameter for conditional variance of representation
        state_cvr, t1_accuracy_cvr, t2_accuracy_cvr = tu.model_selection(cnn, train_data, vali_data, test1_data, test2_data, 
                                                                        num_epochs, learning_rate, batch_size, num_batches, 
                                                                        c, d, ls, key, (28, 28, 1), method="CVR")
        results["CVR"]["test1"].append(t1_accuracy_cvr)
        results["CVR"]["test2"].append(t2_accuracy_cvr)
        
    ############################## SUMMARIZE RESULTS ##############################
    print("\n###########################################################################\n")

    print("NON-REGULARIZED NON-ROTATED TEST ACCURACY = " + str(np.average(results["NO-REG"]["test1"])))
    print("CVP NON-ROTATED TEST ACCURACY = " + str(np.average(results["CVP"]["test1"])))
    print("CVR NON-ROTATED TEST ACCURACY = " + str(np.average(results["CVR"]["test1"])))

    print("\nNON-REGULARIZED ROTATED TEST ACCURACY = " + str(np.average(results["NO-REG"]["test2"])))
    print("CVP ROTATED TEST ACCURACY = " + str(np.average(results["CVP"]["test2"])))
    print("CVR ROTATED TEST ACCURACY = " + str(np.average(results["CVR"]["test2"])))

    logging.info("\n###########################################################################\n")

    logging.info("NON-REGULARIZED NON-ROTATED TEST ACCURACY = " + str(np.average(results["NO-REG"]["test1"])))
    logging.info("CVP NON-ROTATED TEST ACCURACY = " + str(np.average(results["CVP"]["test1"])))
    logging.info("CVR NON-ROTATED TEST ACCURACY = " + str(np.average(results["CVR"]["test1"])))

    logging.info("\nNON-REGULARIZED ROTATED TEST ACCURACY = " + str(np.average(results["NO-REG"]["test2"])))
    logging.info("CVP ROTATED TEST ACCURACY = " + str(np.average(results["CVP"]["test2"])))
    logging.info("CVR ROTATED TEST ACCURACY = " + str(np.average(results["CVR"]["test2"])))
   