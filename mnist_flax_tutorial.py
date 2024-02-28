"""
The following script is currently mostly a copy-paste of the MNIST flax tutorial: https://flax.readthedocs.io/en/latest/quick_start.html
The code is being adjusted in order to reproduce the MNIST part of https://arxiv.org/abs/1710.11469
TODO:
- implement conditional variance regularization including mini batch allocation of (y,id) groups"""

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


class CNN(nn.Module):
    """A simple CNN model."""
    # TODO: might need to use "setup" method instead for an encoder architecture: https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=16, kernel_size=(5, 5), strides=2)(x)
        x = nn.relu(x)
        # unclear whether Heinze uses avg or max pool or any pooling at all
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
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


@jax.jit
def train_step(state, images, labels):
    """Train for a single step."""

    def loss_fn(params_):

        logits = state.apply_fn({'params': params_}, images)

        # TODO: include conditional variance penalty
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=labels).mean()

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state


@jax.jit
def compute_metrics(state, images, labels):

    logits = state.apply_fn({'params': state.params}, images)

    # TODO: include conditional variance penalty
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean()

    # can't find any documentation on what "single_from_model_output" does except for literal source code
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=labels, loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)

    return state

#@jax.jit
def get_grouped_batches(x, y, id, batch_size, rng):
    # draw a random id wo replacement out of the 10000, add it to the badge and keep count of the badge size. 
    # => split how does one propagate the rng?
    # TODO: implement this function
    pass


@jax.jit
def pred_step(state, images):
    logits = state.apply_fn({'params': state.params}, images)
    return logits.argmax(axis=1)


if __name__ == "__main__":

    ################## DEFINE FREE PARAMETES  ##################
    num_epochs = 6
    batch_size = 120
    learning_rate = 0.01
    # number of data points to be augmented by rotation
    c = 200
    # number of original data points in training set, such that number of data points in final training set after augmentaiton is n + c.
    n = 10000

    ################## MNIST DATA AUGMENTATION ##################
    print("\n #################### AUGMENTING MNIST DATA #################### \n")

    # TODO: implement test set 1 and test set 2
    # TODO: properly handle the JAX rngs
    # TODO: data augmentation takes way too long, either write out the data and read it in or improve efficiency
    (x_train, y_train), (x_test1, y_test) = keras.datasets.mnist.load_data()

    x_train = jnp.array(x_train) / 255
    x_test1 = jnp.array(x_test1) / 255
    x_train = jnp.reshape(x_train, (60000, 28, 28, 1))
    x_test1 = jnp.reshape(x_test1, (10000, 28, 28, 1))

    y_train = jnp.array(y_train).astype(jnp.int32)
    y_test = jnp.array(y_test).astype(jnp.int32)

    rng = jax.random.key(0)
    indices = jax.random.choice(rng, jnp.arange(60000), shape=(n,), replace=False)

    x_train = x_train[indices, :, :, :]
    y_train = y_train[indices]
    
    rng = jax.random.key(1)
    aug_indices = jax.random.choice(rng, jnp.arange(10000), shape=(c,), replace=False)
    rng = jax.random.key(2)
    rot_samples = jax.random.choice(rng, jnp.array([35, 70]), shape=(c,), replace=True)

    # list that indexed at relevant id provides list of the indices of all data points with that id
    # note the id of the original data points is set to their index
    id_to_idx = [[i] for i in range(n)]

    for cnt, i in enumerate(aug_indices):

        new_img = ndimage.rotate(x_train[i, :, :, :], rot_samples[i], reshape=False)
        new_img = jnp.reshape(new_img, (1, 28, 28, 1))
        x_train = jnp.vstack((x_train, new_img))

        # add index to the relevant id
        id_to_idx[i].append(n + cnt)

    y_train = jnp.hstack((y_train, y_train[aug_indices]))

    # two test sets will be used to evaluate domain shift invariance: test set 1 is the original MNIST,
    # test set 2 contains the same images but rotated by 35 or 70 degrees with uniform probability
    rng = jax.random.key(3)
    rot_samples = jax.random.choice(rng, jnp.array([35, 70]), shape=(10000,), replace=True)
    x_test2 = x_test1

    # TODO: vectorize this
    for i in range(10000):
        new_img = ndimage.rotate(x_test1[i, :, :, :], rot_samples[i], reshape=False)
        x_test2 = x_test2.at[i, :, :, :].set(new_img)

    ################## TRAINING ##################
    print("\n #################### AUGMENTING MNIST DATA #################### \n")

    tf.random.set_seed(0)
    cnn = CNN()
    rng = jax.random.key(4)
    steps_per_epoch = jnp.ceil(jnp.shape(y_train)[0] / batch_size)
    steps_per_epoch = int(steps_per_epoch)

    metrics_history = {'train_loss': [],
                       'train_accuracy': [],
                       'test_loss': [],
                       'test_accuracy': []}

    for step in range(num_epochs * steps_per_epoch):

        i = step % steps_per_epoch

        train_images = x_train[i*batch_size:(i+1)*batch_size, :, :, :]
        train_labels = y_train[i*batch_size:(i+1)*batch_size]

        state = train_step(state, train_images, train_labels)
        state = compute_metrics(state, train_images, train_labels)

        if (step + 1) % steps_per_epoch == 0:

            # again: cant find any info on what metrics.compute() actually does except source code.
            # from source code I gather that it performs averaging s.t. it averages over the metrics
            # of all batches in that epoch
            for metric, value in state.metrics.compute().items():
                metrics_history[f'train_{metric}'].append(value)

            # reset train_metrics for next training epoch
            state = state.replace(metrics=state.metrics.empty())

            # Compute metrics on the test set after each training epoch
            # need to make a copy of the current training state because the saved metrics will be overwritten
            test_state = state
            test_state = compute_metrics(test_state, x_test, y_test)

            for metric, value in test_state.metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)

            print(f"train epoch: {(step+1) // steps_per_epoch}, "
                  f"loss: {metrics_history['train_loss'][-1]}, "
                  f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")

            print(f"test epoch: {(step+1) // steps_per_epoch}, "
                  f"loss: {metrics_history['test_loss'][-1]}, "
                  f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")

    ################## PLOT LEARNING CURVE ##################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.set_title('Loss')
    ax2.set_title('Accuracy')

    for dataset in ('train', 'test'):
        ax1.plot(metrics_history[f'{dataset}_loss'], label=f'{dataset}_loss')
        ax2.plot(metrics_history[f'{dataset}_accuracy'],
                 label=f'{dataset}_accuracy')

    ax1.legend()
    ax2.legend()
    plt.show()
    plt.clf()

    pred = pred_step(state, x_test)
