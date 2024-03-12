"""
The following script implements conditional variance regularization for domain shift robustness on the MNIST 
data set in Jax/Flax. It reproduces the MNIST results of the following original paper introducing the method: 

https://arxiv.org/abs/1710.11469.

See README for details.
................

TODO:
[- save utils in different python files (after entire code including celeb is finished)]
[- write readme (after entire code including celeb is finished)]
"""

import tensorflow as tf
from flax import linen as nn
import jax
import jax.numpy as jnp
from clu import metrics
from flax.training import train_state
from flax import struct
import optax
import matplotlib.pyplot as plt
import keras
from scipy import ndimage
import numpy as np
from functools import partial
import os
import logging

logging.basicConfig(level=logging.INFO, filename=".\logfile.txt", filemode="w+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


class CNN(nn.Module):
    """A simple CNN model."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        # extract the learned representation and return it separately. This is needed for CVR regularization
        r = x
        #x = nn.Dense(features=32)(x)
        #x = nn.Dense(features=16)(x)
        x = nn.Dense(features=10)(x)
        return x, r


@struct.dataclass
class Metrics(metrics.Collection):
    '''Metrics class is the attribute of the training state that saves the accuracy and the loss'''
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainState(train_state.TrainState):
    """training state consist of: a forward pass function, model parameters, an optimizer and metrics"""
    metrics: Metrics


def create_train_state(module, rng, learning_rate):
    """Creates an initial `TrainState`."""
    # mnist is 28x28
    params = module.init(rng, jnp.ones([1, 28, 28, 1]))['params']

    tx = optax.adam(learning_rate)

    return TrainState.create(apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty())


@partial(jax.jit, static_argnums=(3, 4, 5))
def train_step(state, images, labels, d, l, method="CVP"):
    '''
    Maps a training state, training features, training labels to the new training state after a single gradient descent step

    Parameters:
        state (TrainState): the current training state consisting of parameters, a forward pass function, an optimizer and metrics
        images (jnp.Array): MNIST trainig image batch of shape (<batch_size>, 28, 28, 1). The last 2*d samples MUST be from dublettes
                            where consecutive samples are from the same dublette. All samples before are singlettes
        labels (jnp.Array): MNIST label batch of shape (<batch_size>,). The last 2*d samples MUST be from dublettes
                            where consecutive samples are from the same dublette. All samples before are singlettes
        d (int): the number of (ID, Y) groups with cardinality bigger 1
        l (float): conditional variance regularization parameter
        method (string): regularization method to be applied, either "CVP" or "CVR" for conditional variance of prediction or representation

    Returns:
        state (TrainState): new updated training state after performing one gradient descent step on the provided batch
    '''

    if method not in ["CVP", "CVR"]:
        raise ValueError("Provided method nor recognized. Method should be either CVP or CVR")

    # m is the number of unique (ID, Y) groups in the batch, meaning both singletts (group of cardinality 1)
    # and dublettes (group of cardinality 2). d is the number of unique dublettes in the batch. The number of singletts is hence
    # batch_size - 2*d
    m = jnp.shape(images)[0] - d

    # number of consecutive singleton entries, which is batch_isze - 2*d or m -d
    n_t = m - d

    def loss_fn(params_):

        # forward pass
        logits, repr = state.apply_fn({'params': params_}, images)

        # initialize regularization term
        C = 0

        # regularization contribution of singlettes is 0. Hence, only looking at dublettes, one can gather the dublette info
        # from the last 2d samples in the data
        for i in range(d):

            # get indices of samples in the same dublette
            idxs = jnp.array([n_t + 2*i, n_t + 2*i + 1])

            # calculate variance of the logits inside the dublette
            if method == "CVP":
                vars = jnp.nanvar(jnp.take(logits, idxs, axis=0), axis=0)
            else:
                vars = jnp.nanvar(jnp.take(repr, idxs, axis=0), axis=0)

            C = C + jnp.sum(vars)

        # average over all (ID, Y) groups
        C = C/m

        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean() + l * C

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state


@partial(jax.jit, static_argnums=(3, 4, 5))
def compute_metrics(state, images, labels, d, l, method="CVP"):
    '''
    Given a training state and some features, labels, returns a new training state whose metrics have been updated according to the
    provided data.

    Parameters:
        state (TrainState): the current training state consisting of parameters, a forward pass function, an optimizer and metrics
        images (jnp.Array): MNIST imagees of shape (<n>, 28, 28, 1). The last 2*d samples MUST be from dublettes
                            where consecutive samples are from the same dublette. All samples before are singlettes
        labels (jnp.Array): MNIST labels  of shape (<batch_size>,). The last 2*d samples MUST be from dublettes
                            where consecutive samples are from the same dublette. All samples before are singlettes
        d (int): the number of (ID, Y) groups with cardinality bigger 1
        l (float): conditional variance regularization parameter
        method (string): regularization method to be applied, either "CVP" or "CVR" for conditional variance of prediction or representation

    Returns:
        state (TrainState): new updated training state which contains the calculated metrics in its Metrics attribute
    '''


    # m is the number of unique (ID, Y) groups in the batch, meaning both singletts (group of cardinality 1)
    # and dublettes (group of cardinality 2). d is the number of unique dublettes in the batch. The number of singletts is hence
    # batch_size - 2*d
    m = jnp.shape(images)[0] - d

    # number of consecutive singleton entries, which is batch_isze - 2*d or m -d
    n_t = m - d

    # forward pass
    logits, repr = state.apply_fn({'params': state.params}, images)

    # initialize regularization term
    C = 0

    # regularization contribution of singlettes is 0. Hence, only look at last 2d entries corresponding to dublettes
    for i in range(d):

        # get indices of samples in the same dublette
        idxs = jnp.array([n_t + 2*i, n_t + 2*i + 1])

        # calculate variance of the logits inside the dublettes
        if method == "CVP":
            vars = jnp.nanvar(jnp.take(logits, idxs, axis=0), axis=0)
        else:
            vars = jnp.nanvar(jnp.take(repr, idxs, axis=0), axis=0)

        C = C + jnp.sum(vars)

    # average over all (ID, Y) groups
    C = C/m

    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean() + l*C

    # can't find any documentation on what "single_from_model_output" does except for literal source code
    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=labels, loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)

    return state


@partial(jax.jit, static_argnums=(6, 7, 8))
def get_grouped_batches(x, y, x_orig, y_orig, x_aug, key, batch_size, num_batches, d):
    '''
    Given singlett features/labels, original features/labels from dubletts and augmented features from dublettes, return
    the batches required to run one epoch of conditional variance regularization (CVR).

    Detailed explanation: CVR training requires that each (ID, Y) group is fully contained in exactly one batch. The following
    function assigns exactly d dublettes to each batch. First <batch_size - 2d> datapoints in batch are singletts. 
    Last 2*d data points are from dubletts, s.t. consecutive groups of 2 belong to  same dublette.

    Based on batch_size, num_batches and d some singlett and dublett data is not put in any output batch because batches
    that can only be partially filled are discarded.

    Parameters:
        x (jnp.Array): full MNIST training singlette features of shape (n-c, 28, 28 ,1), where n is the number 
                       of original training data points, c is the number of original data points that were augmented to dublettes by rotation

        y (jnp.Array): full MNIST training singlette labels of shape (n-c,), where n is the number 
                       of original training data points, c is the number of original data points that were augmented to dublettes by rotation
        x_orig (jnp.Array): MNIST non-rotated training features of shape (c, 28, 28, 1) corresponding to the original c data points that were chosen
                            to be augmented by rotation
        y_orig (jnp.Array): MNIST training labels of shape (c, 28, 28, 1) corresponding to the original c data points that were chosen
                            to be augmented by rotation
        x_aug (jnp.Array): MNIST rotated training features of shape (c, 28, 28, 1) corresponding to the rotated versions of the images in x_orig
        key (PRNGKey): PRNG key for data permutation
        batch_size (int): number of data points per output batch
        num_batches (int): number of output batches
        d (int): the number of dublette groups per batch

    Returns:
        x_batches (jnp.Array): array of shape (num_batches, batch_size, 28, 28, 1) containing the images for each batch
        y_batches (jnp.Array): array of shape (num_batches, batch_size) containing the labels for each batch

    '''

    # number of consecutive singlett datapoints per batch. Factor 2 from fact that each dublette contains 2 data points
    n_t = batch_size - 2*d

    # randomly permutate the singlett and dublette data
    key, subkey = jax.random.split(key)
    idxs = jax.random.permutation(subkey, jnp.shape(x)[0])
    x_perm = jnp.take(x, idxs, axis=0)
    y_perm = jnp.take(y, idxs, axis=0)

    key, subkey = jax.random.split(key)
    idxs = jax.random.permutation(subkey, jnp.shape(x_orig)[0])
    x_orig_perm = jnp.take(x_orig, idxs, axis=0)
    x_aug_perm = jnp.take(x_aug, idxs, axis=0)
    y_orig_perm = jnp.take(y_orig, idxs, axis=0)

    # initialize batch output
    x_batches = jnp.zeros((num_batches, batch_size, 28, 28, 1))
    y_batches = jnp.zeros((num_batches, batch_size))

    for i in range(num_batches):
        # fill the first n_t entries of the batch with data points from singlett
        x_batches = x_batches.at[i, :n_t, :, :, :].set(x_perm[i*n_t:(i+1)*n_t, :, :, :])
        y_batches = y_batches.at[i, :n_t].set(y_perm[i*n_t:(i+1)*n_t])

        for j in range(d):
            # first add the original data point
            x_batches = x_batches.at[i, n_t + 2*j, :,:, :].set(x_orig_perm[d*i + j, :, :, :])
            # then add the augmented data point directly afterward
            x_batches = x_batches.at[i, n_t +(2*j)+1, :, :, :].set(x_aug_perm[d*i + j, :, :, :])

            y_batches = y_batches.at[i, n_t + 2*j].set(y_orig_perm[d*i + j])
            y_batches = y_batches.at[i, n_t +(2*j)+1].set(y_orig_perm[d*i + j])

    return x_batches, y_batches.astype(jnp.int32)


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


def train_cnn(train_data, vali_data, num_epochs, learning_rate, batch_size, num_batches, c, d, l, key, method="CVP", tf_seed=0):
    '''
    Given data, all relevant learning parameters, and a regularization method, returns a list containing the training state of 
    each epoch and the epoch that achieved the best validation score.

    Parameters:
        train_data (dic): dictionary with keys "sing_features", "sing_labels", "dub_orig_featrues", "dub_aug_features", "dub_labels".
                          here "sing" and "dub" refers to singlett and dublette groups, where a singlett is defined as a (Y, ID) group only 
                          containing a single datapoint, and a dublette a group containing exactly two datapoints, namely the original one
                          and the augmented one. 
        vali_data (dic): dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        num_epochs (int): number of training epochs
        learning_rate (float): learning rate
        batch_size (int): batch size
        num_batches (int): number of batches
        c (int): number of augmented mnist data points
        d (int): number of dublettes to be contained in each training batch
        l (float): regularization parameter
        key (jax.RNG): jax RNG key
        method (string): regularization method, either "CVP" or "CVR" for conditional variance of prediction and representation respectively
        tf_seed (int): tensorflow rng seed

    Returns:
        states (list of TrainStates): list containing the training states after each epoch
        best_epoch (int): epoch with the lowest validation loss
        best_accuracy (float): accuracy of the epoch with the lowest validation loss
    '''

    if method not in ["CVP", "CVR"]:
        raise ValueError("Provided method nor recognized. Method should be either CVP or CVR")

    logging.info(f"\n#################### START TRAINING {method} l = {l} ####################\n")
    print(f"\n#################### START TRAINING {method} l = {l} ####################\n")

    tf.random.set_seed(tf_seed)
    cnn = CNN()
    key, subkey = jax.random.split(key)
    state = create_train_state(cnn, subkey, learning_rate)

    metrics_history = {'train_loss': [], 'train_accuracy': [], 'vali_loss': [],
                       'vali_accuracy': [], 'test1_loss': [], 'test1_accuracy': [],
                       'test2_loss': [], 'test2_accuracy': []}

    states = []
    for i in range(num_epochs):

        key, subkey = jax.random.split(key)
        x_batches, y_batches = get_grouped_batches(train_data["sing_features"], train_data["sing_labels"],
                                                   train_data["dub_orig_features"], train_data["dub_labels"],
                                                   train_data["dub_aug_features"], subkey, batch_size, num_batches, d)

        for j in range(num_batches):
            train_images = x_batches[j]
            train_labels = y_batches[j]

            state = train_step(state, train_images, train_labels, d, l, method)
            state = compute_metrics(state, train_images, train_labels, d, l, method)

        #  metrics.compute() performs averaging s.t. it averages over the metrics of all batches in that epoch
        for metric, value in state.metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)

        # reset train_metrics for next training epoch
        state = state.replace(metrics=state.metrics.empty())
        states.append(state)

        # Compute metrics on the vali set after each training epoch
        # make copy of  current training state because  saved metrics will be overwritten
        vali_state = state
        vali_state = compute_metrics(vali_state, vali_data["features"], vali_data["labels"], d=c, l=l)

        for metric, value in vali_state.metrics.compute().items():
            metrics_history[f'vali_{metric}'].append(value)


        logging.info(f"train epoch: {i}, loss: {metrics_history['train_loss'][-1]}, accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        logging.info(f"vali epoch: {i}, loss: {metrics_history['vali_loss'][-1]}, accuracy: {metrics_history['vali_accuracy'][-1] * 100}")
        logging.info("\n############################################################# \n")
        print(f"train epoch: {i}, loss: {metrics_history['train_loss'][-1]}, accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        print(f"vali epoch: {i}, loss: {metrics_history['vali_loss'][-1]}, accuracy: {metrics_history['vali_accuracy'][-1] * 100}")
        print("\n############################################################# \n")

    ################## PLOT AND SAVE LEARNING CURVE ##################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.set_title('CE Loss')
    ax2.set_title('Accuracy')

    dic = {'train': 'train', 'vali': 'validation'}
    for dataset in ('train', 'vali'):
        ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dic[dataset]}')
        ax2.plot(metrics_history[f'{dataset}_accuracy'], label=f'{dic[dataset]}')

    ax1.set_xlabel("epoch")
    ax2.set_xlabel("epoch")

    ax1.set_xticks(np.arange(num_epochs, step=2))
    ax2.set_xticks(np.arange(num_epochs, step=2))
    ax1.legend()
    ax2.legend()
    ax1.grid(True)
    ax2.grid(True)

    lr_str = str(learning_rate).replace(".", ",")
    l_str = str(l).replace(".", ",")
    plt.savefig(f".\learning_curves\learning_curve_{method}_lr{lr_str}_l{l_str}_e{num_epochs}_bs{batch_size}.png")
    plt.clf()

    best_epoch = max(enumerate(metrics_history['vali_accuracy']), key=lambda x: x[1])[0]
    best_accuracy = max(metrics_history['vali_accuracy'])

    logging.info(f"best vali epoch: {best_epoch}")
    logging.info(f"best vali accuracy: {best_accuracy}")

    print(f"best vali epoch: {best_epoch}")
    print(f"best vali accuracy: {best_accuracy}")

    return states, best_epoch, best_accuracy


def model_selection(train_data, vali_data, test1_data, test2_data, num_epochs, learning_rate, batch_size, num_batches,
                    c, d, ls, key, method="CVP", tf_seed=0):
    '''
    Given data, all relevant learning parameters, a regularization method and a list of regularization parameters to be validated
    returns the final training state of the model that achieves the best validation score.

    Parameters:
        train_data (dic): dictionary with keys "sing_features", "sing_labels", "dub_orig_featrues", "dub_aug_features", "dub_labels".
                          here "sing" and "dub" refers to singlett and dublette groups, where a singlett is defined as a (Y, ID) group only 
                          containing a single datapoint, and a dublette a group containing exactly two datapoints, namely the original one
                          and the augmented one. 
        vali_data (dic): dictionary with keys "featrues" and "labels", values are jax arrays containing the data
        test1_data (dic): dictionary with keys "featrues" and "labels", containing non-rotated, non-augmented test data
        test2_data (dic): dictionary with keys "featrues" and "labels", containing rotated, non-augmented test data
        num_epochs (int): number of training epochs
        learning_rate (float): learning rate
        batch_size (int): batch size
        num_batches (int): number of batches
        c (int): number of augmented mnist data points
        d (int): number of dublettes to be contained in each training batch
        ls (list): list of regularization parameters to be validated and selected
        key (jax.RNG): jax RNG key
        method (string): regularization method, either "CVP" or "CVR" for conditional variance of prediction and representation respectively
        tf_seed (int): tensorflow rng seed

    Returns:
        state (TrainState): state of the selected model
        test1_accuracy (float): accuracy on test set 1 of the best model. Test set 1 contains non-rotated images only
        test2_accuracy (float:) accuracy on test set 2 of the best model. Test set 2 contains rotated images only
    '''

    if method not in ["CVP", "CVR"]:
        raise ValueError("Provided method nor recognized. Method should be either CVP or CVR")

    logging.info(f"#################### PERFORMING {method} MODEL SELECTION FOR l = {ls} ####################")
    print(f"#################### PERFORMING {method} MODEL SELECTION FOR l = {ls} ####################")

    dic = {}
    best_l = None
    best_accuracy = -1000

    for l in ls:
        key, subkey = jax.random.split(key)
        states, epoch, accuracy = train_cnn(train_data, vali_data, num_epochs, learning_rate, batch_size,
                                            num_batches, c, d, l, subkey, method, tf_seed)

        if accuracy > best_accuracy:
            best_l = l
            best_accuracy = accuracy

        dic[str(l)] = {"epoch": None, "accuracy": None, "states": states}
        dic[str(l)]["epoch"] = epoch
        dic[str(l)]["accuracy"] = accuracy

    best_epoch = dic[str(best_l)]["epoch"]
    logging.info("\n#############################################################\n")
    logging.info(f"THE BEST REGULRAIZATION PARAMETER IS {best_l} AT EPOCH {best_epoch} WITH VALI ACCURACY {best_accuracy}")
    print("\n#############################################################\n")
    print(f"THE BEST REGULRAIZATION PARAMETER IS {best_l} AT EPOCH {best_epoch} WITH VALI ACCURACY {best_accuracy}")

    state = dic[str(best_l)]["states"][best_epoch]
    test1_state = state
    test2_state = state

    test1_state = compute_metrics(test1_state, test1_data["features"], test1_data["labels"], d=0, l=best_l)
    test2_state = compute_metrics(test2_state, test2_data["features"], test2_data["labels"], d=0, l=best_l)

    test1_accuracy = test1_state.metrics.compute()["accuracy"]
    test2_accuracy = test2_state.metrics.compute()["accuracy"]

    logging.info(f"\nACHIEVED NON-ROTATED {method} TEST ACCURACY: {test1_accuracy}")
    logging.info(f"\nACHIEVED ROTATED {method} TEST ACCURACY: {test2_accuracy}")
    print(f"\nACHIEVED NON-ROTATED {method} TEST ACCURACY: {test1_accuracy}")
    print(f"\nACHIEVED ROTATED {method} TEST ACCURACY: {test2_accuracy}")

    return state, test1_accuracy, test2_accuracy


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
    ls = [0.01, 0.1, 1]

    seed = 6542

    ################## LOAD/CREATE DATA ##################
    key = jax.random.key(seed)
    train_data, vali_data, test1_data, test2_data = load_aug_mnist(c, seed, n)

    ################## TRAIN MODELS ##################

    # run unregularized case as model selection with only l=0 to choose from, method chosen does not matter for l=0
    state, t1_accuracy, t2_accuracy = model_selection(train_data, vali_data, test1_data, test2_data, num_epochs, 
                                                      learning_rate, batch_size,num_batches, c, d, [0], key, 
                                                      method="CVP", tf_seed=0)

    # select regularization parameter for conditional variance of prediction
    state_cvp, t1_accuracy_cvp, t2_accuracy_cvp = model_selection(train_data, vali_data, test1_data, test2_data, num_epochs, 
                                                                  learning_rate, batch_size, num_batches, c, d, ls, key, 
                                                                  method="CVP", tf_seed=0)

    # select regularization parameter for conditional variance of representation
    state_cvr, t1_accuracy_cvr, t2_accuracy_cvr = model_selection(train_data, vali_data, test1_data, test2_data,num_epochs, 
                                                                  learning_rate, batch_size, num_batches, c, d, ls, key, 
                                                                  method="CVR", tf_seed=0)

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
