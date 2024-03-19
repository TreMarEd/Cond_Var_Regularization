"""
Contains all necessary utilities for the training loop using conditional variance regularization.
"""

import tensorflow as tf
import jax
import jax.numpy as jnp
from clu import metrics
from flax.training import train_state
from flax import struct
import optax
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from jax.tree_util import Partial
import logging


logging.basicConfig(level=logging.INFO, filename=".\logfile.txt", filemode="w+",
                    format="%(asctime)-15s %(levelname)-8s %(message)s")


@struct.dataclass
class Metrics(metrics.Collection):
    '''Metrics class is the attribute of the training state that saves the accuracy and the loss'''
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss') # type: ignore


class TrainState(train_state.TrainState):
    """training state consist of: a forward pass function, model parameters, an optimizer and metrics"""
    metrics: Metrics


def create_train_state(module, rng, size_0, size_1, ccs, learning_rate):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, size_0, size_1, ccs]))['params']

    tx = optax.adam(learning_rate)

    return TrainState.create(apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty())


@Partial(jax.jit, static_argnums=(3, 4, 5))
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


@Partial(jax.jit, static_argnums=(3, 4, 5))
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

#TODO: find out why this crashes sometimes without error warning
#@Partial(jax.jit, static_argnums=(5, 6, 7, 9, 10, 11))
def get_grouped_batches(x, y, x_orig, y_orig, x_aug, size_0, size_1, ccs, key, batch_size, num_batches, d):
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
    x_batches = jnp.zeros((num_batches, batch_size, size_0, size_1, ccs))
    y_batches = jnp.zeros((num_batches, batch_size))

    for i in range(num_batches):
        print(f"CURRENT BATCH {i}")
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


def train_cnn(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, learning_rate, batch_size, num_batches, c_vali, d, l, 
              key, size_0, size_1, ccs, method="CVP", tf_seed=0):
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
    key, subkey = jax.random.split(key)
    state = create_train_state(cnn, subkey, size_0, size_1, ccs, learning_rate)

    metrics_history = {'train_loss': [], 'train_accuracy': [], 'vali_loss': [],
                       'vali_accuracy': [], 'test1_loss': [], 'test1_accuracy': [],
                       'test2_loss': [], 'test2_accuracy': []}

    states = []

    print("GETTING GROUPED BATCHES")
    x_batches, y_batches = get_grouped_batches(train_data["sing_features"], train_data["sing_labels"],
                                                   train_data["dub_orig_features"], train_data["dub_labels"],
                                                   train_data["dub_aug_features"], size_0, size_1, ccs, subkey, batch_size, num_batches, d)
    print("DONE GETTING GROUPED BATCHES")
    
    for i in range(num_epochs):

        key, subkey = jax.random.split(key)
        """
        x_batches, y_batches = get_grouped_batches(train_data["sing_features"], train_data["sing_labels"],
                                                   train_data["dub_orig_features"], train_data["dub_labels"],
                                                   train_data["dub_aug_features"], size_0, size_1, ccs, subkey, batch_size, num_batches, d)
        """
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


        logging.info(f"train epoch: {i}, loss: {metrics_history['train_loss'][-1]}, accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        logging.info(f"vali epoch: {i}, loss: {metrics_history['vali_loss'][-1]}, accuracy: {metrics_history['vali_accuracy'][-1] * 100}")
        logging.info(f"test1 epoch: {i}, loss: {metrics_history['test1_loss'][-1]}, accuracy: {metrics_history['test1_accuracy'][-1] * 100}")
        logging.info(f"test2 epoch: {i}, loss: {metrics_history['test2_loss'][-1]}, accuracy: {metrics_history['test2_accuracy'][-1] * 100}")
        logging.info("\n############################################################# \n")
        print(f"train epoch: {i}, loss: {metrics_history['train_loss'][-1]}, accuracy: {metrics_history['train_accuracy'][-1] * 100}")
        print(f"vali epoch: {i}, loss: {metrics_history['vali_loss'][-1]}, accuracy: {metrics_history['vali_accuracy'][-1] * 100}")
        print(f"test1 epoch: {i}, loss: {metrics_history['test1_loss'][-1]}, accuracy: {metrics_history['test1_accuracy'][-1] * 100}")
        print(f"test2 epoch: {i}, loss: {metrics_history['test2_loss'][-1]}, accuracy: {metrics_history['test2_accuracy'][-1] * 100}")
        
        print("\n############################################################# \n")

    ################## PLOT AND SAVE LEARNING CURVE ##################
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.set_title('CE Loss')
    ax2.set_title('Accuracy')

    dic = {'train': 'train', 'vali': 'validation', 'test1': 'test1', 'test2': 'test2'}
    for dataset in ('train', 'vali', 'test1', 'test2'):
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
    best_accuracy = metrics_history['vali_accuracy'][best_epoch]

    return states, best_epoch, best_accuracy


def model_selection(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, learning_rate, batch_size, num_batches,
                    c_vali, d, ls, key, size_0, size_1, ccs, method="CVP", tf_seed=0):
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
        states, epoch, accuracy = train_cnn(cnn, train_data, vali_data, test1_data, test2_data, num_epochs, 
                                            learning_rate, batch_size, num_batches, c_vali, d, l, subkey, 
                                            size_0, size_1, ccs, method, tf_seed)

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

    logging.info(f"\nACHIEVED NON-SHIFTED {method} TEST ACCURACY: {test1_accuracy}")
    logging.info(f"\nACHIEVED SHIFTED {method} TEST ACCURACY: {test2_accuracy}")
    print(f"\nACHIEVED NON-SHIFTED {method} TEST ACCURACY: {test1_accuracy}")
    print(f"\nACHIEVED SHIFTED {method} TEST ACCURACY: {test2_accuracy}")

    return state, test1_accuracy, test2_accuracy
