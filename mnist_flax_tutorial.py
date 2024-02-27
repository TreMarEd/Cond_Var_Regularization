"""
The following script is a copy-paste of the MNIST flax tutorial: https://flax.readthedocs.io/en/latest/quick_start.html
Some personal comments were added for understanding.
TODO:
- use keras jnp version of mnist to make data augmentation easier
- implement Heinze data augmentation
- implement conditional variance regularization
"""

import tensorflow as tf
import tensorflow_datasets as tfds
from flax import linen as nn
import jax
import jax.numpy as jnp
from clu import metrics
from flax.training import train_state
from flax import struct
import optax
import matplotlib.pyplot as plt
import keras


"""
def get_datasets(num_epochs, batch_size):

    train_ds = tfds.load('mnist', split='train', data_dir='.')
    test_ds = tfds.load('mnist', split='test')

    # normalize train set
    train_ds = train_ds.map(lambda sample: {'image': tf.cast(
        sample['image'], tf.float32) / 255., 'label': sample['label']})

    # normalize test set
    test_ds = test_ds.map(lambda sample: {'image': tf.cast(
        sample['image'], tf.float32) / 255., 'label': sample['label']})

    # create shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
    # shuffle(n) reads in the first n data points, then chooses one uniformly at random, and replaces the chosen
    # data point with data point n+1 and repeats.
    train_ds = train_ds.repeat(num_epochs).shuffle(1024)
    test_ds = test_ds.shuffle(1024)

    # group into batches of batch_size and skip incomplete batch, prefetch the next sample to improve latency
    train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

    return train_ds, test_ds
"""


class CNN(nn.Module):
    """A simple CNN model."""
    # TODO: might need to use "setup" method instead for an encoder architecture: https://flax.readthedocs.io/en/latest/guides/flax_fundamentals/flax_basics.html
    # TODO: implement Heinze-Deml architecture
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        # flatten for classification layer
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
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

# uncomment jit to see values while debugging


@jax.jit
def train_step(state, images, labels):
    """Train for a single step."""

    def loss_fn(params_):

        # unclear why one needs to pass the params as single key dictionary, can't find anything in the internet
        # batch_size * 10 array
        logits = state.apply_fn({'params': params_}, images)

        # TODO: include conditional variance penalty
        # scalar
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean()

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state

# uncomment jit to see values while debugging


@jax.jit
def compute_metrics(state, images, labels):

    # unclear why one needs to pass the params as single key dictionary, can't find anything in the internet
    # batch_size x 10 array
    logits = state.apply_fn({'params': state.params}, images)

    # TODO: include conditional variance penalty
    # scalar
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels).mean()

    # can't find any documentation on what "single_from_model_output" does except for literal source code
    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=labels, loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)

    return state


@jax.jit
def pred_step(state, images):
    logits = state.apply_fn({'params': state.params}, images)
    return logits.argmax(axis=1)


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # TODO: shuffle the data
    x_train = jnp.array(x_train) / 255
    x_test = jnp.array(x_test) / 255
    x_train = jnp.reshape(x_train, (60000, 28, 28, 1))
    x_test = jnp.reshape(x_test, (10000, 28, 28, 1))

    y_train = jnp.array(y_train).astype(jnp.int32)
    y_test = jnp.array(y_test).astype(jnp.int32)

    
    

    num_epochs = 8
    batch_size = 32
    learning_rate = 0.005

    # train_ds, test_ds = get_datasets(num_epochs, batch_size)

    tf.random.set_seed(0)
    init_rng = jax.random.key(0)
    cnn = CNN()
    state = create_train_state(cnn, init_rng, learning_rate)

    steps_per_epoch = jnp.ceil(jnp.shape(y_train)[0] / batch_size)
    steps_per_epoch = int(steps_per_epoch)

    # metrics_history will contain the relevant metrics after each training epoch
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
            # test set is also fed in batches, I think this should also work by just feeding the entire thing at once
            # for test_batch in test_ds.as_numpy_iterator():
            #    test_state = compute_metrics(state=test_state, batch=test_batch)

            for metric, value in test_state.metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)

            print(f"train epoch: {(step+1) // steps_per_epoch}, "
                  f"loss: {metrics_history['train_loss'][-1]}, "
                  f"accuracy: {metrics_history['train_accuracy'][-1] * 100}")

            print(f"test epoch: {(step+1) // steps_per_epoch}, "
                  f"loss: {metrics_history['test_loss'][-1]}, "
                  f"accuracy: {metrics_history['test_accuracy'][-1] * 100}")

    # Plot loss and accuracy in subplots
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
