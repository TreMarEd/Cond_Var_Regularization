"""
The following script is currently mostly a copy-paste of the MNIST flax tutorial: https://flax.readthedocs.io/en/latest/quick_start.html
The code is being adjusted in order to reproduce the MNIST part of https://arxiv.org/abs/1710.11469
TODO:
- implement conditional variance regularization """

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
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
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


#@jax.jit
def train_step(state, images, labels, ids, l):
    """Train for a single step."""
    # id deterministically determiney y for MNIST, so the (y, id) groups are equivalent to id groups
    unique_ids = jnp.unique(ids)
    m = len(unique_ids)

    def loss_fn(params_):

        logits = state.apply_fn({'params': params_}, images)
        pred_logits = logits.max(axis=1)

        # m is the number of different id groups in the batch
        
        vars = jnp.empty(m)

        for i, id in enumerate(unique_ids):
            # contrary to jax documentation, jnp.where returns a tuple, which needs to be converted to an jnp.array for jnp.take, 
            # which needs to be jnp.squeezed to have no redundant dimensions
            idxs =  jnp.squeeze(jnp.array(jnp.where(ids==id)))
            var = jnp.nanvar(jax.numpy.take(pred_logits, idxs))
            vars = vars.at[i].set(var)
        
        C = jnp.mean(vars)

        # TODO: include conditional variance penalty
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean() + l * C     

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state


#@jax.jit
def compute_metrics(state, images, labels, ids, l):
    unique_ids = jnp.unique(ids)
    m = len(unique_ids)
    logits = state.apply_fn({'params': state.params}, images)
    pred_logits = logits.max(axis=1)

    # m is the number of different id groups in the batch
    vars = jnp.empty(m)

    for i, id in enumerate(unique_ids):
        # contrary to jax documentation, jnp.where returns a tuple, which needs to be converted to an jnp.array for jnp.take, 
        # which needs to be jnp.squeezed to have no redundant dimensions
        idxs =  jnp.squeeze(jnp.array(jnp.where(ids==id)))
        var = jnp.nanvar(jax.numpy.take(pred_logits, idxs))
        vars = vars.at[i].set(var)
        
    C = jnp.mean(vars)

    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean() + l*C

    # can't find any documentation on what "single_from_model_output" does except for literal source code
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=labels, loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)

    return state

# TODO: find out how to jit this without error
#@partial(jax.jit, static_argnums=3)
def get_grouped_batches(x, y, id_to_idx, batch_size, key):

    num_batches = jnp.floor(len(id_to_idx)/batch_size)
    num_batches = num_batches.astype(int)
    ids = [i for i in range(len(id_to_idx))]

    x_batches = []
    y_batches = []
    id_batches = []

    for i in jnp.arange(num_batches):

        x_batch = []
        y_batch = []
        id_batch = []

        key, subkey = jax.random.split(key)
        sample_ids = np.array(jax.random.choice(
            subkey, jnp.array(ids), shape=(batch_size,), replace=False))
        
        for id in sample_ids:
            for idx in id_to_idx[id]:
                x_batch.append(jnp.reshape(x[idx, :, :, :], (1, 28, 28, 1)))
                y_batch.append(y[idx])
                id_batch.append(id)

        ids = list(set(ids) - set(sample_ids))

        x_batch = jnp.vstack(x_batch)
        y_batch = jnp.hstack(y_batch)
        id_batch = jnp.hstack(id_batch)

        x_batches.append(x_batch)
        y_batches.append(y_batch)
        id_batches.append(id_batch)

    return x_batches, y_batches, id_batches


@jax.jit
def pred_step(state, images):
    logits = state.apply_fn({'params': state.params}, images)
    return logits.argmax(axis=1)


if __name__ == "__main__":

    ################## DEFINE FREE PARAMETES  ##################
    num_epochs = 20
    batch_size = 120
    learning_rate = 0.007
    # regularization parameter
    l = 100
    seed = 2134
    # number of data points to be augmented by rotation
    c = 200
    # number of original data points in training set, such that number of data points in final training set after augmentaiton is n + c.
    n = 10000

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
    indices = jax.random.choice(
        subkey, jnp.arange(60000), shape=(n,), replace=False)

    x_train = x_train[indices, :, :, :]
    y_train = y_train[indices]

    key, subkey = jax.random.split(key)
    aug_indices = jax.random.choice(
        subkey, jnp.arange(10000), shape=(c,), replace=False)

    key, subkey = jax.random.split(key)
    rot_samples = jax.random.uniform(
        subkey, shape=(c,), minval=35., maxval=70.)

    # list that indexed at relevant id provides list of the indices of all data points with that id
    # note the id of the original data points is set to their index
    id_to_idx = [[i] for i in range(n)]

    for cnt, i in enumerate(aug_indices):

        new_img = ndimage.rotate(
            x_train[i, :, :, :], rot_samples[i], reshape=False)
        new_img = jnp.reshape(new_img, (1, 28, 28, 1))
        x_train = jnp.vstack((x_train, new_img))

        # add index to the relevant id
        id_to_idx[i].append(n + cnt)

    y_train = jnp.hstack((y_train, y_train[aug_indices]))

    # two test sets will be used to evaluate domain shift invariance: test set 1 is the original MNIST,
    # test set 2 contains the same images but rotated by 35 or 70 degrees with uniform probability
    key, subkey = jax.random.split(key)
    rot_samples = jax.random.uniform(
        subkey, shape=(10000,), minval=35., maxval=70.)

    x_test2 = x_test1

    # TODO: vectorize this for efficiency
    for i in range(10000):
        new_img = ndimage.rotate(
            x_test1[i, :, :, :], rot_samples[i], reshape=False)
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
        x_batches, y_batches, id_batches = get_grouped_batches(
            x_train, y_train, id_to_idx, batch_size, subkey)

        for j in range(len(x_batches)):
            train_images = x_batches[j]
            train_labels = y_batches[j]
            train_ids = id_batches[j]

            state = train_step(state, train_images, train_labels, train_ids, l)
            state = compute_metrics(state, train_images, train_labels, train_ids, l)

        # again: cant find any info on what metrics.compute() actually does except source code.
        # from source code I gather that it performs averaging s.t. it averages over the metrics
        # of all batches in that epoch
        for metric, value in state.metrics.compute().items():
            metrics_history[f'train_{metric}'].append(value)

            # reset train_metrics for next training epoch
            state = state.replace(metrics=state.metrics.empty())

        # Compute metrics on the test set after each training epoch
        # need to make a copy of the current training state because the saved metrics will be overwritten
        
        #TODO: need to pass ids of test data: trivial as no non-trivial groups are contained
        test1_state = state
        test1_state = compute_metrics(test1_state, x_test1, y_test, ids=jnp.arange(10000), l=l)

        test2_state = state
        test2_state = compute_metrics(test2_state, x_test2, y_test, ids=jnp.arange(10000), l=l)

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

    plt.savefig(".\learning_curve.png")

    plt.show()
    plt.clf()

    pred = pred_step(state, x_test2)
