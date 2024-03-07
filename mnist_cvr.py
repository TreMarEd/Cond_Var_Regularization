"""
The following script implements conditional variance regularization (CVR) for domain shift robustness on the MNIST 
data set in Jax/Flax. It reproduces the MNIST results of the following original paper introducing CVR: 

https://arxiv.org/abs/1710.11469.

See README for details.
................

TODO:
- implement trian vali test split for regulaization selection, including a function to be called to train for a specific parameter
- implement separate function for mnis augmentation
- assert correct combinations of batch_size num_batches and d. Provide guidelines
- write readme
- write requirements.txt
- try cond var of repr
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



class CNN(nn.Module):
    """A simple CNN model."""
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        # x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=10)(x)
        return x


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
    _params = module.init(rng, jnp.ones([1, 28, 28, 1]))['params']

    tx = optax.adam(learning_rate)

    return TrainState.create(apply_fn=module.apply, params=_params, tx=tx, metrics=Metrics.empty())


@partial(jax.jit, static_argnums=(3, 4))
def train_step(state, images, labels, d, l):
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

    Returns:
        state (TrainState): new updated training state after performing one gradient descent step on the provided batch
    '''

    # m is the number of unique (ID, Y) groups in the batch, meaning both singletts (group of cardinality 1)
    # and dublettes (group of cardinality 2). d is the number of unique dublettes in the batch. The number of singletts is hence
    # batch_size - 2*d
    m = jnp.shape(images)[0] - d

    # number of consecutive singleton entries, which is batch_isze - 2*d or m -d
    n_t = m - d

    def loss_fn(params_):

        # forward pass
        logits = state.apply_fn({'params': params_}, images)

        # initialize regularization term
        C = 0

        # regularization contribution of singlettes is 0. Hence, only looking at dublettes, one can gather the dublette info
        # from the last 2d samples in the data
        for i in range(d):

            # get indices of samples in the same dublette
            idxs = jnp.array([n_t + 2*i, n_t + 2*i + 1])

            # calculate variance of the logits inside the dublette
            vars = jnp.nanvar(jnp.take(logits, idxs, axis=0), axis=0)

            C = C + jnp.sum(vars)

        # average over all (ID, Y) groups
        C = C/m

        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels).mean() + l * C

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state


@partial(jax.jit, static_argnums=(3, 4))
def compute_metrics(state, images, labels, d, l):
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
    logits = state.apply_fn({'params': state.params}, images)

    # initialize regularization term
    C = 0

    # regularization contribution of singlettes is 0. Hence, only look at last 2d entries corresponding to dublettes
    for i in range(d):

        # get indices of samples in the same dublette
        idxs = jnp.array([n_t + 2*i, n_t + 2*i + 1])

        # calculate variance of the logits inside the dublette
        vars = jnp.nanvar(jnp.take(logits, idxs, axis=0), axis=0)

        C = C + jnp.sum(vars)

    # average over all (ID, Y) groups
    C = C/m

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean() + l*C

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
            x_batches = x_batches.at[i, n_t + 2*j, :, :, :].set(x_orig_perm[d*i + j, :, :, :])
            # then add the augmented data point directly afterward
            x_batches = x_batches.at[i, n_t + (2*j)+1, :, :, :].set(x_aug_perm[d*i + j, :, :, :])

            y_batches = y_batches.at[i, n_t + 2*j].set(y_orig_perm[d*i + j])
            y_batches = y_batches.at[i, n_t + (2*j)+1].set(y_orig_perm[d*i + j])

    return x_batches, y_batches.astype(jnp.int32)


@jax.jit
def pred_step(state, images):
    logits = state.apply_fn({'params': state.params}, images)
    return logits.argmax(axis=1)


def create_aug_mnist(c, seed):
    """
    """

    print("\n #################### AUGMENTING MNIST DATA #################### \n")

    n = 10000

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
    aug_indices = jax.random.choice(subkey, jnp.arange(10000), shape=(c,), replace=False)

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

    base_path = f".\\augmented_mnist\\seed{seed}_c{c}"

    if not os.path.exists(base_path):
        os.makedirs(f"{base_path}\\train")
        os.makedirs(f"{base_path}\\vali")
        os.makedirs(f"{base_path}\\test")
        
    
    jnp.save(base_path + "\\train\\x_train_sing.npy", x_train_sing)
    jnp.save(base_path + "\\train\\y_train_sing.npy", y_train_sing)
    jnp.save(base_path + "\\train\\x_train_orig.npy", x_train_orig)
    jnp.save(base_path + "\\train\\y_train_orig.npy", y_train_orig)
    jnp.save(base_path + "\\train\\x_train_aug.npy", x_train_aug)

    jnp.save(base_path + "\\vali\\x_vali_sing.npy", x_vali_sing)
    jnp.save(base_path + "\\vali\\y_vali_sing.npy", y_vali_sing)
    jnp.save(base_path + "\\vali\\x_vali_orig.npy", x_vali_orig)
    jnp.save(base_path + "\\vali\\y_vali_orig.npy", y_vali_orig)
    jnp.save(base_path + "\\vali\\x_vali_aug.npy", x_vali_aug)

    jnp.save(base_path + "\\test\\x_test1.npy", x_test1)
    jnp.save(base_path + "\\test\\x_test2.npy", x_test2)
    jnp.save(base_path + "\\test\\y_test.npy", y_test)
    
    return None


if __name__ == "__main__":

    train_data, vali_data, test1_data, test2_data = create_aug_mnist(200, 234)
    
    ################## DEFINE FREE PARAMETES  ##################
    num_epochs = 18
    batch_size = 102
    # number of data points to be augmented by rotation
    c = 200
    # number of original data points in training set
    n = 10000
    # number of dublett groups per batch
    d = int(np.ceil(c/np.floor((n+c)/batch_size)))
    num_batches = int(np.floor(c/d))
    learning_rate = 0.005
    # regularization parameter
    l = 5
    seed = 234
    
    ################## MNIST DATA AUGMENTATION ##################
    print("\n #################### AUGMENTING MNIST DATA #################### \n")

    (x_train, y_train), (x_test1, y_test) = keras.datasets.mnist.load_data()

    x_train = jnp.array(x_train) / 255
    x_test1 = jnp.array(x_test1) / 255
    x_train = jnp.reshape(x_train, (60000, 28, 28, 1))
    x_test1 = jnp.reshape(x_test1, (10000, 28, 28, 1))

    y_train = jnp.array(y_train).astype(jnp.int32)
    y_test = jnp.array(y_test).astype(jnp.int32)

    key = jax.random.key(seed)
    key, subkey = jax.random.split(key)
    indices = jax.random.choice(subkey, jnp.arange(60000), shape=(n,), replace=False)

    x_train = x_train[indices, :, :, :]
    y_train = y_train[indices]

    key, subkey = jax.random.split(key)
    aug_indices = jax.random.choice(subkey, jnp.arange(10000), shape=(c,), replace=False)

    x_orig = x_train[aug_indices, :, :, :]
    y_orig = y_train[aug_indices]

    x = jnp.delete(x_train, aug_indices, axis=0)
    y = jnp.delete(y_train, aug_indices, axis=0)

    key, subkey = jax.random.split(key)
    rot_samples = jax.random.uniform(subkey, shape=(c,), minval=35., maxval=70.)

    x_aug = jnp.zeros(jnp.shape(x_orig))

    for i in range(c):
        new_img = ndimage.rotate(
            x_orig[i, :, :, :], rot_samples[i], reshape=False)
        x_aug = x_aug.at[i, :, :, :].set(new_img)

    # two test sets will be used to evaluate domain shift invariance: test set 1 is the original MNIST,
    # test set 2 contains the same images but rotated by 35 or 70 degrees with uniform probability
    key, subkey = jax.random.split(key)
    rot_samples = jax.random.uniform(subkey, shape=(10000,), minval=35., maxval=70.)

    x_test2 = x_test1

    for i in range(10000):
        new_img = ndimage.rotate(x_test1[i, :, :, :], rot_samples[i], reshape=False)
        x_test2 = x_test2.at[i, :, :, :].set(new_img)

    ################## TRAINING ##################
    print("\n #################### START TRAINING #################### \n")

    tf.random.set_seed(0)
    cnn = CNN()
    key, subkey = jax.random.split(key)
    state = create_train_state(cnn, subkey, learning_rate)

    metrics_history = {'train_loss': [], 'train_accuracy': [], 'test1_loss': [],
                       'test1_accuracy': [], 'test2_loss': [], 'test2_accuracy': []}

    for i in range(num_epochs):

        key, subkey = jax.random.split(key)
        x_batches, y_batches = get_grouped_batches(x, y, x_orig, y_orig, x_aug, key, batch_size, num_batches, d)

        for j in range(num_batches):
            train_images = x_batches[j]
            train_labels = y_batches[j]

            state = train_step(state, train_images, train_labels, d, l)
            state = compute_metrics(state, train_images, train_labels, d, l)

        #  metrics.compute() performs averaging s.t. it averages over the metrics of all batches in that epoch
        for metric, value in state.metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)

            # reset train_metrics for next training epoch
            state = state.replace(metrics=state.metrics.empty())

        # Compute metrics on the test sets after each training epoch
        # make copy of  current training state because  saved metrics will be overwritten
        test1_state = state
        test1_state = compute_metrics(test1_state, x_test1, y_test, d=0, l=l)

        test2_state = state
        test2_state = compute_metrics(test2_state, x_test2, y_test, d=0, l=l)

        for metric, value in test1_state.metrics.compute().items():
            metrics_history[f'test1_{metric}'].append(value)

        for metric, value in test2_state.metrics.compute().items():
            metrics_history[f'test2_{metric}'].append(value)

        print(f"train epoch: {i+1}, "f"loss: {metrics_history['train_loss'][-1]}, "f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        print(f"test1 epoch: {i+1}, "f"loss: {metrics_history['test1_loss'][-1]}, "f"accuracy: {metrics_history['test1_accuracy'][-1] * 100}")
        print(f"test2 epoch: {i+1}, "f"loss: {metrics_history['test2_loss'][-1]}, "f"accuracy: {metrics_history['test2_accuracy'][-1] * 100}")
        print("\n############################################################# \n")

    ################## PLOT LEARNING CURVE ##################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.set_title('CE Loss')
    ax2.set_title('Accuracy')

    dic = {'train': 'train', 'test1': 'orignal MNIST test','test2': 'rotated MNIST test'}
    for dataset in ('train', 'test1', 'test2'):
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
    plt.savefig(f".\learning_curves\learning_curve_lr{lr_str}_l{l_str}_e{num_epochs}_bs{batch_size}.png")

    plt.show()
    plt.clf()

    pred = pred_step(state, x_test2)
