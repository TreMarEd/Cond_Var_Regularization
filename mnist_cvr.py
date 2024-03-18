"""
The following script implements conditional variance regularization for domain shift robustness on the MNIST 
data set in Jax/Flax. It reproduces the MNIST results of the following original paper introducing the method: 

https://arxiv.org/abs/1710.11469.

See README for details.
................

TODO:
- pack functions usable for both MNIST and CelebA into a utils directory: these include:
    - Metrics, TrainState, create_train_state, train_step, compute_metrics, get_grouped_batch, train_cnn, model_selection
[- write readme (after entire code including celeb is finished)]
[- new requirements.txt]
"""

from flax import linen as nn
import jax
import jax.numpy as jnp
import keras
from scipy import ndimage
import os
import logging
import train_utils as tu

logging.basicConfig(level=logging.INFO, filename=".\logfile.txt", filemode="w+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


class CNN_mnist(nn.Module):
    """A simple CNN model."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        # extract the learned representation and return it separately. This is needed for CVR regularization
        r = x
        x = nn.Dense(features=10)(x)
        return x, r


def create_aug_mnist(c, seed, n):
    '''
    Given the number of data points to be augmented (c) and an RNG seed returns and saves augmented MNIST data. The data is augmented
    by randomly selecting <c> data points and rotation them by an angle uniformly distributed in [35, 70] degrees.
    Both training and validation data contain c augmented data points.
    The test1 data consists of the standard domain, where NO image is rotated
    The test2 data consists of the rotated domain, where ALL images are rotated.

    Parameters:
        c (int): number of datapoints to be augmenteed by rotation
        seed (int): rng seed to be used
        n (int): number of original data points to use in the training and validation set


    Returns:
        train_data (dic): dictionary with keys "sing_features", "sing_labels", "dub_orig_featrues", "dub_aug_features", "dub_labels".
                          here "sing" and "dub" refers to singlett and dublette groups, where a singlett is defined as a (Y, ID) group only 
                          containing a single datapoint, and a dublette a group containing exactly two datapoints, namely the original one
                          and the augmented one. 
        vali_data (dic): dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        test1_data (dic): dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        test2_data (dic): dictionary with keys "featrues" and "labels", values are jax arrays containing the data

    '''

    logging.info("\n#################### AUGMENTING MNIST DATA #################### \n")
    print("\n#################### AUGMENTING MNIST DATA #################### \n")

    (x, y), (x_test1, y_test) = keras.datasets.mnist.load_data()

    x = jnp.array(x) / 255
    x_test1 = jnp.array(x_test1) / 255

    x = jnp.reshape(x, (60000, 28, 28, 1))
    x_test1 = jnp.reshape(x_test1, (10000, 28, 28, 1))

    y = jnp.array(y).astype(jnp.int32)
    y_test = jnp.array(y_test).astype(jnp.int32)

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, jnp.arange(60000), shape=(2*n,), replace=False)

    x_train = x[indices[:n], :, :, :]
    y_train = y[indices[:n]]

    x_vali = x[indices[n:], :, :, :]
    y_vali = y[indices[n:]]

    key, subkey = jax.random.split(key)
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

    base_path = f".\\augmented_mnist\\seed{seed}_c{c}_n{n}"

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
    The data is augmentedby randomly selecting <c> data points and rotation them by an angle uniformly distributed 
    in [35, 70] degrees. Both training and validation data contain c augmented data points.
    The test1 data consists of the standard domain, where NO image is rotated
    The test2 data consists of the rotated domain, where ALL images are rotated.

    Parameters:
        c (int): number of datapoints to be augmenteed by rotation
        seed (int): rng seed to be used
        n (int): number of original data points to use in the training and validation set

    Returns:
        train_data (dic): dictionary with keys "sing_features", "sing_labels", "dub_orig_featrues", "dub_aug_features", "dub_labels".
                          here "sing" and "dub" refers to singlett and dublette groups, where a singlett is defined as a (Y, ID) group only 
                          containing a single datapoint, and a dublette a group containing exactly two datapoints, namely the original one
                          and the augmented one. 
        vali_data (dic): dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        test1_data (dic): dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        test2_data (dic): dictionary with keys "featrues" and "labels", values are jax arrays containing the data

    '''

    base_path = f".\\augmented_mnist\\seed{seed}_c{c}_n{n}"

    if not os.path.exists(base_path):
        return create_aug_mnist(c, seed, n)

    else:
        logging.info("\n#################### LOADING MNIST DATA #################### \n")
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
    num_epochs = 30
    learning_rate = 0.003

    # n is the number of original data points in training set. Needs to be an integer multiple of 100 
    # for below parameter choices to be optimal: batches are sorted such that always the last 2*d datapoints
    # are augmented groups, meaning d consecutive pairs with first the original data point, 
    # then the augmented/rotated datapoint. The below choice of batch size, d and number of batches ensures that
    # the final batch is not partially filled
    n = 10000

    # fractions of data points to be augmented, should be an integer multiple of 0.01 for below parameter choice to be optimal
    alpha = 0.02

    # number of data points to be augmented by rotation, equal to number of dublette groups in the final data set
    c = int(alpha * n)
    batch_size = int((1 + alpha) * 100)
    # d is the number of dublette (Y, ID) groups per batch
    d = int(alpha * 100)
    num_batches = int(n / 100)

    # regularization parameters on which to perform model selection
    ls = [0.01, 0.1, 1, 10]

    seed = 6542

    ################## LOAD/CREATE DATA ##################
    key = jax.random.key(seed)
    train_data, vali_data, test1_data, test2_data = load_aug_mnist(c, seed, n)

    ################## TRAIN MODELS ##################
    cnn = CNN_mnist()
    # run unregularized case as model selection with only l=0 to choose from, method chosen does not matter for l=0
    state, t1_accuracy, t2_accuracy = tu.model_selection(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, 
                                                      learning_rate, batch_size, num_batches, c, d, [0], key,
                                                      size_0=28, size_1=28, method="CVP", tf_seed=0)

    # select regularization parameter for conditional variance of prediction
    state_cvp, t1_accuracy_cvp, t2_accuracy_cvp = tu.model_selection(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, 
                                                                  learning_rate, batch_size, num_batches, c, d, ls, key, 
                                                                  size_0=28, size_1=28, method="CVP", tf_seed=0)
    
    # select regularization parameter for conditional variance of representation
    state_cvr, t1_accuracy_cvr, t2_accuracy_cvr = tu.model_selection(cnn, train_data, vali_data, test1_data, test2_data,num_epochs, 
                                                                  learning_rate, batch_size, num_batches, c, d, ls, key, 
                                                                  size_0=28, size_1=28, method="CVR", tf_seed=0)
    
    logging.info("\n###########################################################################\n")

    logging.info(f"NON-REGULARIZED NON-ROTATED TEST ACCURACY = {t1_accuracy}")
    logging.info(f"CVP NON-ROTATED TEST ACCURACY = {t1_accuracy_cvp}")
    logging.info(f"CVR NON-ROTATED TEST ACCURACY = {t1_accuracy_cvr}")

    logging.info(f"\nNON-REGULARIZED ROTATED TEST ACCURACY = {t2_accuracy}")
    logging.info(f"CVP ROTATED TEST ACCURACY = {t2_accuracy_cvp}")
    logging.info(f"CVR ROTATED TEST ACCURACY = {t2_accuracy_cvr}")
    
    print("\n###########################################################################\n")

    print(f"NON-REGULARIZED NON-ROTATED TEST ACCURACY = {t1_accuracy}")
    print(f"CVP NON-ROTATED TEST ACCURACY = {t1_accuracy_cvp}")
    print(f"CVR NON-ROTATED TEST ACCURACY = {t1_accuracy_cvr}")

    print(f"\nNON-REGULARIZED ROTATED TEST ACCURACY = {t2_accuracy}")
    print(f"CVP ROTATED TEST ACCURACY = {t2_accuracy_cvp}")
    print(f"CVR ROTATED TEST ACCURACY = {t2_accuracy_cvr}")
