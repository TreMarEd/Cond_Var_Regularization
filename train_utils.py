"""
Author: Marius Tresoldi, spring 2024

The following script implements some training utilities to implement conditional variance regularization 
in jax/flax for domain shift robust transfer learning in computer vision. The idea of conditional variance regularization is
introduced in the following paper:

https://arxiv.org/abs/1710.11469

The general training loop structure is pretty much copy-pasted from the following MNIST flax tutorial:

https://flax.readthedocs.io/en/latest/experimental/nnx/mnist_tutorial.html

In the following, (ID, Y) groups that only contain a single data point will be called singletts, while (ID, Y) groups containing
exactly two datapoints through data augmentation will be called dubletts.
"""

#TODO: remove image_shape as function inpu as it can always be inferred from other inputs

import tensorflow as tf
import jax
import jax.numpy as jnp
from clu import metrics
from flax.training import train_state
from flax import struct
import optax
from jax.tree_util import Partial


@struct.dataclass
class Metrics(metrics.Collection):
    '''Metrics class is the attribute of the training state that saves the accuracy and the loss'''
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss') # type: ignore


class TrainState(train_state.TrainState):
    """training state consist of: a forward pass function, model parameters, an optimizer and metrics"""
    metrics: Metrics


def create_train_state(module, rng, img_shape, learning_rate):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, *img_shape]))['params']
    tx = optax.adam(learning_rate)

    return TrainState.create(apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty())


@Partial(jax.jit, static_argnums=(3, 4, 5))
def train_step(state, images, labels, d, l, method="CVR"):
    '''
    Maps a training state, training images, training labels to the new training state after a single gradient descent step

    Parameters:
        state (TrainState): the current training state consisting of parameters, a forward pass function, an optimizer and 
                            metrics
        images (jnp.Array): MNIST trainig image batch of shape (<batch_size>, <size_0>, <size_1>, <color channels>). 
                            The last 2*d samples MUST be from dublettes where consecutive samples are from the same dublette. 
                            All samples before are singlettes
        labels (jnp.Array): MNIST label batch of shape (<batch_size>,). The last 2*d samples MUST be from dublettes
                            where consecutive samples are from the same dublette. All samples before are singlettes
        d (int):            number of dublettes per batch
        l (float):          conditional variance regularization parameter
        method (string):    regularization method to be applied, either "CVP" or "CVR" for conditional variance of prediction 
                            or representation

    Returns:
        state (TrainState): new updated training state after performing one gradient descent step on the provided batch
    '''

    if method not in ["CVP", "CVR"]:
        raise ValueError("Provided method not recognized. Method should be either CVP or CVR")

    # m is the number of unique (ID, Y) groups in the batch, meaning both singletts and dublettes. d is the number of 
    # unique dublettes in the batch. The number of singletts is hence batch_size - 2*d
    m = jnp.shape(images)[0] - d

    # number of consecutive singleton entries, which is batch_size - 2*d or m -d
    n_t = m - d

    def loss_fn(params):
        logits, repr = state.apply_fn({'params': params}, images)
        # initialize regularization term
        C = 0

        # regularization contribution of singlettes is 0. Hence, only looking at dublettes, one can gather the dublette info
        # from the last 2d samples in the data
        for i in range(d):
            # get indices of samples in the same dublette
            idxs = jnp.array([n_t + 2*i, n_t + 2*i + 1])

            # calculate variance of the logits (CVP) or representations (CVR) inside the dublette
            if method == "CVP":
                vars = jnp.nanvar(jnp.take(logits, idxs, axis=0), axis=0)
            else:
                vars = jnp.nanvar(jnp.take(repr, idxs, axis=0), axis=0)

            C = C + jnp.sum(vars)

        # average over all (ID, Y) groups
        C = C / m
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels).mean() + l * C

        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)

    return state


@Partial(jax.jit, static_argnums=(3, 4, 5))
def compute_metrics(state, images, labels, d, l, method="CVR"):
    '''
    Given a training state and some features, labels, returns a new training state whose metrics have been updated according to the
    provided data.

    Parameters:
        state (TrainState): the current training state consisting of parameters, a forward pass function, an optimizer and 
                            metrics
        images (jnp.Array): MNIST trainig image batch of shape (<batch_size>, <size_0>, <size_1>, <color channels>). 
                            The last 2*d samples MUST be from dublettes where consecutive samples are from the same dublette. 
                            All samples before are singlettes
        labels (jnp.Array): MNIST label batch of shape (<batch_size>,). The last 2*d samples MUST be from dublettes
                            where consecutive samples are from the same dublette. All samples before are singlettes
        d (int):            number of dublettes per batch
        l (float):          conditional variance regularization parameter
        method (string):    regularization method to be applied, either "CVP" or "CVR" for conditional variance of prediction 
                            or representation

    Returns:
        state (TrainState): new updated training state which contains the calculated metrics in its Metrics attribute
    '''

    # m is the number of unique (ID, Y) groups in the batch, meaning both singletts and dublettes. d is the number of 
    #unique dublettes in the batch. The number of singletts is hence batch_size - 2*d
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

    metric_updates = state.metrics.single_from_model_output(logits=logits, labels=labels, loss=loss)
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)

    return state


#TODO: when jitted this function terminated the program without any error or warning. This is impossible as it only
#consists of a for loop with no statement that could terminate the program. This might acutally be a jax bug
#@Partial(jax.jit, static_argnums=(5, 6, 7, 9, 10, 11))
def get_grouped_batches(x, y, x_orig, y_orig, x_aug, img_shape, key, batch_size, num_batches, d):
    '''
    Given singlett features/labels, original features/labels from dubletts and augmented features from dublettes, return
    the batches required to run one epoch of conditional variance regularization (CVR).

    Detailed explanation: CVR training requires that each (ID, Y) group is fully contained in exactly one batch. The following
    function assigns exactly d dublettes to each batch. First <batch_size - 2d> datapoints in batch are singletts. 
    Last 2*d data points are from dubletts, s.t. datapoints of same dublette are consecutive to each othre.

    Note that because every batch needs to contain exactly d dublettes, it is possible that some data provided to the function
    is not put in any batch in order to satisfy this constraint. Hence, one should optimize the choice of batch_size, num_batches
    d, and c, where c is the number of augmented data points. 
    F.e. one can set batch_size to 100+d and choose c such that batch_size = (n+c)/(100+d) is an integer.

    Parameters:
        x (jnp.Array):      training singlette features of shape (n-c, *img_shape), where n is the number 
                            of original training data points, c is the number of original data points that were augmented 
                            to dublettes 

        y (jnp.Array):      training singlette labels of shape (n-c,), where n is the number 
                            of original training data points, c is the number of original data points that were augmented to dublettes
        x_orig (jnp.Array): non-augmented training features of shape (c, *img_shape) corresponding to the original c data points 
                            that were chosen to be augmented
        y_orig (jnp.Array): training labels of shape (c, *img_shape) corresponding to the original c data points that were chosen
                            to be augmented
        x_aug (jnp.Array):  augmented training features of shape (c, img_shape) corresponding to the augmentd versions of the 
                            images in x_orig
        img_shape (tuple):  tuple describing the shape of a single input image
        key (PRNGKey):      PRNG key for data permutation
        batch_size (int):   number of data points per output batch
        num_batches (int):  number of output batches
        d (int):            the number of dublette groups per batch

    Returns:
        x_batches (jnp.Array): array of shape (num_batches, batch_size, *img_shape) containing the images for each batch
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
    x_batches = jnp.zeros((num_batches, batch_size, img_shape[1], img_shape[0], img_shape[2]))
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


def train_cnn(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, learning_rate, batch_size, num_batches, c_vali, d,
              l, key, img_shape, method="CVR", tf_seed=0):
    '''
    Given data, all relevant learning parameters and a regularization method, returns the training state with the best
    validation accuracy, all relevant accuracies (vali, test1, test2) and saves the learning curve.

    Parameters:
        cnn (flax module):      cnn model to be trained
        train_data (dic):       dictionary with keys "sing_features", "sing_labels", "dub_orig_featrues", "dub_aug_features", 
                                "dub_labels"."sing" and "dub" refers to singlett and dublette groups. values are jax arrays 
                                containing the data
        vali_data (dic):        dictionary with keys "features" and "labels", values are jax arrays containing the vali data
        test1_data (dic):       dictionary with keys "features" and "labels", values are jax arrays containing the test1 data
        test2_data (dic):       dictionary with keys "features" and "labels", values are jax arrays containing the test2 data
        num_epochs (int):       number of training epochs
        learning_rate (float):  learning rate
        batch_size (int):       batch size
        num_batches (int):      number of batches
        c_vali (int):           number of augmented mnist data points in the vali set. This information is needed to efficiently
                                calculate the regularization for the vali set, as the last 2*c data points will be dublettes
        d (int):                number of dublettes to be contained in each training batch
        l (float):              regularization parameter
        key (jax.RNG):          jax RNG key
        img_shape (tuple):      tuple describing the shape of a single input image
        method (string):        regularization method, either "CVP" or "CVR" for conditional variance of prediction and
                                representation respectively
        tf_seed (int):          tensorflow rng seed

    Returns:
        best_state (TrainStates):   the training state with the best validation accuracy
        vali_accuracy (float):      validation accuracy of the best state
        test1_accuracy (float):     test 1 accuracy of the best state
        test2_accuracy (float):     test 2 accuracy of the best state
    '''

    if method not in ["CVP", "CVR"]:
        raise ValueError("Provided method not recognized. Method should be either CVP or CVR")

    print(f"\n#################### START TRAINING {method} l = {l} ####################\n")

    tf.random.set_seed(tf_seed)
    key, subkey = jax.random.split(key)
    state = create_train_state(cnn, subkey, img_shape, learning_rate)

    metrics_history = {'train_loss': [], 'train_accuracy': [], 'vali_loss': [],
                       'vali_accuracy': [], 'test1_loss': [], 'test1_accuracy': [],
                       'test2_loss': [], 'test2_accuracy': []}

    states = []

    x_batches, y_batches = get_grouped_batches(train_data["sing_features"], train_data["sing_labels"],
                                               train_data["dub_orig_features"], train_data["dub_labels"],
                                               train_data["dub_aug_features"], img_shape, subkey, batch_size, 
                                               num_batches, d)
        
    for i in range(num_epochs):

        key, subkey = jax.random.split(key)

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
        vali_state = compute_metrics(vali_state, vali_data["features"], vali_data["labels"], d=c_vali, l=l)

        for metric, value in vali_state.metrics.compute().items():
            metrics_history[f'vali_{metric}'].append(value)

        test1_state = state
        test1_state = compute_metrics(test1_state, test1_data["features"], test1_data["labels"], d=0, l=l)

        for metric, value in test1_state.metrics.compute().items():
            metrics_history[f'test1_{metric}'].append(value)
        
        test2_state = state
        test2_state = compute_metrics(test2_state, test2_data["features"], test2_data["labels"], d=0, l=l)

        for metric, value in test2_state.metrics.compute().items():
            metrics_history[f'test2_{metric}'].append(value)

        print(f"train epoch: {i}, loss: {metrics_history['train_loss'][-1]}, accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        print(f"vali epoch: {i}, loss: {metrics_history['vali_loss'][-1]}, accuracy: {metrics_history['vali_accuracy'][-1] * 100}")
        print(f"test1 epoch: {i}, loss: {metrics_history['test1_loss'][-1]}, accuracy: {metrics_history['test1_accuracy'][-1] * 100}")
        print(f"test2 epoch: {i}, loss: {metrics_history['test2_loss'][-1]}, accuracy: {metrics_history['test2_accuracy'][-1] * 100}")
        print("\n############################################################# \n")

    best_epoch = max(enumerate(metrics_history['vali_accuracy']), key=lambda x: x[1])[0]
    best_state = states[best_epoch]

    vali_accuracy = metrics_history['vali_accuracy'][best_epoch]
    test1_accuracy = metrics_history['test1_accuracy'][best_epoch]
    test2_accuracy = metrics_history['test2_accuracy'][best_epoch]

    return best_state, vali_accuracy, test1_accuracy, test2_accuracy


def model_selection(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, learning_rate, batch_size, num_batches,
                    c_vali, d, ls, key, img_shape, method="CVR", tf_seed=0):
    '''
    Given data, all relevant learning parameters, a regularization method and a list of regularization parameters to be validated
    returns the final training state of the model that achieves the best validation score.

    Parameters:
        cnn (flax module):      cnn model to be trained
        train_data (dic):       dictionary with keys "sing_features", "sing_labels", "dub_orig_featrues", "dub_aug_features", 
                                "dub_labels"."sing" and "dub" refers to singlett and dublette groups. values are jax arrays 
                                containing the data
        vali_data (dic):        dictionary with keys "features" and "labels", values are jax arrays containing the vali data
        test1_data (dic):       dictionary with keys "features" and "labels", values are jax arrays containing the test1 data
        test2_data (dic):       dictionary with keys "features" and "labels", values are jax arrays containing the test2 data
        num_epochs (int):       number of training epochs
        learning_rate (float):  learning rate
        batch_size (int):       batch size
        num_batches (int):      number of batches
        c_vali (int):           number of augmented mnist data points in the vali set. This information is needed to efficiently
                                calculate the regularization for the vali set, as the last 2*c data points will be dublettes
        d (int):                number of dublettes to be contained in each training batch
        ls (list):              list of regularization parameters to be validated
        key (jax.RNG):          jax RNG key
        img_shape (tuple):      tuple describing the shape of a single input image
        method (string):        regularization method, either "CVP" or "CVR" for conditional variance of prediction and
                                representation respectively
        tf_seed (int):          tensorflow rng seed

    Returns:
        best_state (TrainState):    state of the selected model
        test1_accuracy (float):     accuracy on test set 1 of the best model
        test2_accuracy (float:)     accuracy on test set 2 of the best model
    '''

    if method not in ["CVP", "CVR"]:
        raise ValueError("Provided method nor recognized. Method should be either CVP or CVR")

    print(f"#################### PERFORMING {method} MODEL SELECTION FOR l = {ls} ####################")

    best_l = None
    best_vali_accuracy = -1000

    for l in ls:
        
        state, vali_accuracy, test1_accuracy, test2_accuracy = train_cnn(cnn, train_data, vali_data, test1_data, test2_data, 
                                                                         num_epochs, learning_rate, batch_size, num_batches, 
                                                                         c_vali, d, l, subkey, img_shape, method, 
                                                                         tf_seed)
        key, subkey = jax.random.split(key)

        if vali_accuracy > best_vali_accuracy:
            best_l = l
            best_state = state
            best_vali_accuracy = vali_accuracy
            best_test1_accuracy = test1_accuracy
            best_test2_accuracy = test2_accuracy
    
    print("\n#############################################################\n")
    print(f"THE BEST REGULRAIZATION PARAMETER IS {best_l} WITH:")
    print(f"\nACHIEVED NON-SHIFTED {method} TEST ACCURACY: {best_test1_accuracy}")
    print(f"\nACHIEVED SHIFTED {method} TEST ACCURACY: {best_test2_accuracy}")

    return best_state, test1_accuracy, test2_accuracy
