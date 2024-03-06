"""
The following script is currently mostly a copy-paste of the MNIST flax tutorial: https://flax.readthedocs.io/en/latest/quick_start.html
The code is being adjusted in order to reproduce the MNIST part of https://arxiv.org/abs/1710.11469
TODO:
- rewrite s.t. everything is jittable
- implement trian vali test split for regulaization selection, including a function to be called to train for a specific parameter
- general cosmetics: comments, docstrings, annotations, modularization
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


class CNN(nn.Module):
    """A simple CNN model."""
    # TODO: might need to use "setup" method instead for an encoder architecture: https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        x = nn.Conv(features=32, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        #x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # flatten for classification layer
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=10)(x)
        return x


# the Metrics class contains all metrics that we wish to save in the training state
@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    # cant find any documentation on clu.metrics.Average.from_output in the internet
    # I assume this method automatically calculates the average loss
    loss: metrics.Average.from_output('loss')

# it is not readable here but the training state will actually consist of:
# a forward pass function, model parameters, an optimizer and values of the metrics chosen above
class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate):
    """Creates an initial `TrainState`."""
    # mnist is 28x28
    _params = module.init(rng, jnp.ones([1, 28, 28, 1]))['params']

    tx = optax.adam(learning_rate)

    return TrainState.create(apply_fn=module.apply, params=_params, tx=tx, metrics=Metrics.empty())


@partial(jax.jit, static_argnums=(3,4))
def train_step(state, images, labels, d, l):
    """Train for a single step."""

    # number of groups in batch
    m = jnp.shape(images)[0] - d
    # number of consecutive singleton entries
    n_t = m - d

    def loss_fn(params_):

        logits = state.apply_fn({'params': params_}, images)

        C = 0

        for i in range(d):
            idxs = jnp.array([n_t + 2*i, n_t + 2*i + 1])
            vars = jnp.nanvar(jnp.squeeze(jnp.take(logits, idxs, axis=0)), axis=0)
            C = C + jnp.sum(vars)
        
        C = C/m
       
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean() + l * C     

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state


@partial(jax.jit, static_argnums=(3,4))
def compute_metrics(state, images, labels, d, l):

    # number of groups in batch
    m = jnp.shape(images)[0] - d

    # number of consecutive singleton entries
    n_t = m - d

    logits = state.apply_fn({'params': state.params}, images)

    C = 0

    for i in range(d):
        idxs = jnp.array([n_t + 2*i, n_t + 2*i + 1])
        vars = jnp.nanvar(jnp.squeeze(jnp.take(logits, idxs, axis=0)), axis=0)
        C = C + jnp.sum(vars)
        
    C = C/m

    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean() + l*C

    # can't find any documentation on what "single_from_model_output" does except for literal source code
    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=labels, loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)

    return state


@partial(jax.jit, static_argnums=(6,7,8))
def get_grouped_batches(x, y, x_orig, y_orig, x_aug, key, batch_size, num_batches, d):

    # number of datapoints per batch that do not belong to a group with more than 1 member (t for trivial)
    # factor 2 comes from fact that each non-trivial group has size 2
    n_t = batch_size - 2*d

    key, subkey = jax.random.split(key)
    idxs = jax.random.permutation(subkey, jnp.shape(x)[0])
    x_perm = jnp.take(x, idxs, axis=0)
    y_perm = jnp.take(y, idxs, axis=0)

    key, subkey = jax.random.split(key)
    idxs = jax.random.permutation(subkey, jnp.shape(x_orig)[0])
    x_orig_perm = jnp.take(x_orig, idxs, axis=0)
    x_aug_perm = jnp.take(x_aug, idxs, axis=0)
    y_orig_perm = jnp.take(y_orig, idxs, axis=0)

    x_batches = jnp.zeros((num_batches, batch_size, 28, 28, 1))
    y_batches = jnp.zeros((num_batches, batch_size))

    '''
    fill the last entries of the batch with the data points form non-trivial groups, where 
    data points of the same group are consecutive
    example: 10'000 original data points, of which 200 get augmented, with a batch size of 120
    this implies 9800 data points that are singletons, and 200 non-singleton groups with 2 members each
    such that one has 10200 data points. with a batchsize of 120 this implies 85 batches, such that
    each batch will contain floor(200/85) = 2 non-singletons, corresponding to 4 data points
    for each batch take 116 data points out of the singletons, and then attach at the end the four
    non-singleton data points, where points in the same group follow each other
    '''

    for i in range(num_batches):
        # fill the first entries of the batch with data points from trivial groups
        x_batches = x_batches.at[i, :n_t, :, :, :].set(x_perm[i*n_t:(i+1)*n_t, :, :, :])
        y_batches = y_batches.at[i, :n_t].set(y_perm[i*n_t:(i+1)*n_t])


        for j in range(d):
            # first add the original data point
            x_batches = x_batches.at[i, n_t + 2*j, :, :, :].set(x_orig_perm[d*i + j, :, :, :])
            # then add the augmented data point directly afterward
            x_batches = x_batches.at[i, n_t +(2*j)+1, :, :, :].set(x_aug_perm[d*i + j, :, :, :])

            y_batches = y_batches.at[i, n_t + 2*j].set(y_orig_perm[d*i + j])
            y_batches = y_batches.at[i, n_t +(2*j)+1].set(y_orig_perm[d*i + j])


    return x_batches, y_batches.astype(jnp.int32)


              


@jax.jit
def pred_step(state, images):
    logits = state.apply_fn({'params': state.params}, images)
    return logits.argmax(axis=1)


if __name__ == "__main__":

    ################## DEFINE FREE PARAMETES  ##################
    num_epochs = 25
    # 102 is ideal for n=10000 and c = 200
    batch_size = 102
    learning_rate = 0.005
    # regularization parameter
    l = 10
    seed = 234
    # number of data points to be augmented by rotation
    c = 200
    # number of original data points in training set, such that number of data points in final training set after augmentaiton is n + c.
    n = 10000
    
    d = int(np.ceil(c/np.floor((n+c)/batch_size)))
    num_batches = int(np.floor(c/d))
    # with the homogeneous distribution of singleotons and dublettes the last badge will be incomplete and needs to be discarded

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
        new_img = ndimage.rotate(x_orig[i, :, :, :], rot_samples[i], reshape=False)
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

        # again: cant find any info on what metrics.compute() actually does except source code.
        # from source code I gather that it performs averaging s.t. it averages over the metrics
        # of all batches in that epoch
        for metric, value in state.metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)

            # reset train_metrics for next training epoch
            state = state.replace(metrics=state.metrics.empty())

        # Compute metrics on the test set after each training epoch
        # need to make a copy of the current training state because the saved metrics will be overwritten
        
        test1_state = state
        test1_state = compute_metrics(test1_state, x_test1, y_test, d=0, l=l)

        test2_state = state
        test2_state = compute_metrics(test2_state, x_test2, y_test, d=0, l=l)

        for metric, value in test1_state.metrics.compute().items():
            metrics_history[f'test1_{metric}'].append(value)

        for metric, value in test2_state.metrics.compute().items():
            metrics_history[f'test2_{metric}'].append(value)

        print(f"train epoch: {i+1}, "f"loss: {metrics_history['train_loss'][-1]}, "
              f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")

        print(f"test1 epoch: {i+1}, "f"loss: {metrics_history['test1_loss'][-1]}, "
              f"accuracy: {metrics_history['test1_accuracy'][-1] * 100}")

        print(f"test2 epoch: {i+1}, "f"loss: {metrics_history['test2_loss'][-1]}, "
              f"accuracy: {metrics_history['test2_accuracy'][-1] * 100}")

        print("\n############################################################# \n")

    ################## PLOT LEARNING CURVE ##################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.set_title('CE Loss')
    ax2.set_title('Accuracy')

    dic = {'train': 'train', 'test1': 'orignal MNIST test',
           'test2': 'rotated MNIST test'}
    for dataset in ('train', 'test1', 'test2'):
        ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dic[dataset]}')
        ax2.plot(metrics_history[f'{dataset}_accuracy'],
                 label=f'{dic[dataset]}')

    ax1.set_xlabel("epoch")
    ax2.set_xlabel("epoch")

    ax1.set_xticks(np.arange(num_epochs, step=2))
    ax2.set_xticks(np.arange(num_epochs, step=2))

    ax1.legend()
    ax2.legend()

    ax1.grid(True)
    ax2.grid(True)

    plt.savefig(".\learning_curve_l0008_lr0004.png")

    plt.show()
    plt.clf()

    pred = pred_step(state, x_test2)
