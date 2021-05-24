""" Sequence for training and hyperparam tuning."""

from tensorflow.keras import callbacks
from data import CifarData
from network import Network
import tqdm
import argparse
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from datetime import datetime as dt
import tensorflow as tf
import os
from pathlib import Path

mirrored_strategy = tf.distribute.MirroredStrategy()

ap = argparse.ArgumentParser()
ap.add_argument("-epochs", "--epochs", type=int, required=True, help="how many epochs to train for")
ap.add_argument("-batchsize", "--batchsize", type=int, required=True, help="what batchsize touse")
args = vars(ap.parse_args())


CKPT_FOLDER = "PLACEHOLDER"
BATCH_SIZE = args['batchsize']
EPOCHS = args['epochs']
LMDA_L1 = .0005
LMDA_L2 = .001
LMBDA_ACTIVITY = None #tf.keras.regularizers.L1(l1 = LMDA_L1)
LMBDA_KERNEL = None #tf.keras.regularizers.L1L2(l1 = LMDA_L1, l2 = LMDA_L2)
LMBDA_BIAS = None #tf.keras.regularizers.L1L2(l1 = LMDA_L1, l2 = LMDA_L2)

def ts_file():
    return dt.now().strftime("%Y-%m-%d_%H-%M-%S.%f")

def ts():
    return dt.now().strftime("%Y-%m-%d %H:%M:%S.%f:")

def coarse_search(lm_min, lm_max, h):
    """
    Coarse search for L2 (Lasso) regulization term.

    Args:
        :param float32 lm_min: Minimum Lambda to search for
        :param float32 lm_max: Maximum Lambda to search for
        :param float32 h: the interval between each lambda step
    """
    print(f"{ts()} I source/model/train.py] Starting training...")
    ffolder_ckpt = Path(CKPT_FOLDER) / f"batch_size_{BATCH_SIZE}" / f"LambdaL1_{LMDA_L1}"
    print(f"{ts()} I source/model/train.py] Built checkpoint folders")

    
    
    fpath_ckpt =  ffolder_ckpt / "cp.ckpt"
    print(f"{ts()} I source/model/train.py] Saving to {fpath_ckpt}")
    
    # ==================================================================
    #                  INIT Classes and Parameters
    # ------------------------------------------------------------------
    n = Network()
    n.construct(LMBDA_KERNEL, LMBDA_BIAS, LMBDA_ACTIVITY)
    cifar = CifarData()
    ckpt = tf.keras.callbacks.ModelCheckpoint(
            filepath = fpath_ckpt, monitor = 'val_loss', mode = 'min',
            verbose = 1, save_best_only = True, save_weights_only = False
    )
    # ==================================================================

    print(f"{ts()} I source/model/train.py] Loading Dataset ...")

    # ==================================================================
    #                  Load The Dataset
    # ------------------------------------------------------------------   
    train_data , val_data = cifar.load_data()   
    x_train, y_train, x_val, y_val = n.shape_td(train_data, val_data)
    # ==================================================================
    
    print(f"{ts()} I source/model/train.py] Dataset Loaded.")
    print(f"{ts()} I soruce/model/train.py] Fitting Model ...")

    # ==================================================================
    #                  Hyperparam Tuning
    # ------------------------------------------------------------------
    
    lgnd = []
    for lm in np.arange(lm_min, lm_max, h):
        ffolder_ckpt = Path(CKPT_FOLDER) / f"batch_size_{BATCH_SIZE}" / f"LambdaL2_{lm}"
        try:
            os.makedirs(ffolder_ckpt)
        except:
            pass
        n.construct(LMBDA_KERNEL, LMBDA_BIAS, lmda_activiy = tf.keras.regularizers.L1(l1 = lm)) 
        h = n.model.fit(x_train, y_train, batch_size = BATCH_SIZE,
            epochs = 10, validation_data = (x_val, y_val),
            callbacks = [ckpt])
        plt.plot(h.history['val_accuracy'])
        lgnd.append(lm)
    # ==================================================================

    plt.title('Val Accuracy for Lambas for 24 epochs')
    plt.legend(lgnd)
    plt.show()

@tf.function
def accuracy(y, logits):
    argP = tf.math.argmax(logits, axis = 0)
    argY = tf.math.argmax(tf.convert_to_tensor(y), axis = 0)
    delta = tf.math.subtract(argP, argY)
    zeros = tf.math.subtract(tf.size(delta), tf.math.count_nonzero(delta, 0, dtype = tf.int32))
    return tf.math.divide(zeros, tf.size(delta))

@tf.function
def train_step(model, x, y, loss_obj, batch_size):
    with tf.GradientTape() as tape:
        logits = model.model(x, training = True)
        loss_value = tf.reduce_sum(loss_obj(y, logits)) * (1. / batch_size)
        grads = tape.gradient(loss_value, model.model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.model.trainable_weights))
        acc = accuracy(y, logits)
    return loss_value, acc

@tf.function
def val_step(model, x, y, loss_obj, batch_size):
    val_logits = model.model(x, training = False)
    val_loss = tf.reduce_sum(loss_obj(y, val_logits)) * (1. / batch_size)
    val_acc = accuracy(y, val_logits)
    return val_loss, val_acc

@tf.function
def float_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def batch_data(x, y, bs):
    """
        Splits the data x and y into batches.

        Args:
            :param x: Training data
            :param y: Training Labels
            :param bs: batch size
        Returns:
            x_batches, y_batches
    """
    x_batches = np.array_split(x, int(len(x) / bs))
    y_batches = np.array_split(y, int(len(x) / bs))
    return x_batches, y_batches

def shuffle_data(x, y, bs):
    """
        Shuffles the training data and splits into new batches.

        Args:
            :param x: Training Data
            :param y: Training Labels
            :param bs: batch size
        Returns:
            x_batches: Training Data shuffled and in batches
            y_batches: Training labels shuffled (same idx as x_batches) and in batches
    """
    print("+++ Shuffled Dataset +++")
    idxes = np.arange(len(x))
    np.random.shuffle(idxes)
    x = np.array(x)[idxes]
    y = np.array(y)[idxes]
    xb, yb = batch_data(x, y, bs)
    return xb, yb

def fit(model, x, y, batch_size, epochs, x_val, y_val, shuffle = True):
    """
        Customer Fitting function for our deep neural network.

        Args:
            :param x: Input training data
            :param y: Input training labels
            :param batch_size: Batch size
            :param epochs: Number of epochs to train for
            :param val_data: (x, y) of the validation dataset
            :param shuffle: If we should shuffle the train data with every epoch
        Returns:
            history: history dict with accuracy and validation data
    """
    x_batches, y_batches = batch_data(x, y, batch_size)
    x_val_batches, y_val_batches = batch_data(x_val, y_val, batch_size)
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
    
    loss_obj = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    for epoch in range(epochs):
        tmp_acc = 0
        tmp_val_acc = 0
        tmp_loss = 0
        tmp_val_loss = 0
        print(f"\nEpoch: {epoch+1} out of {epochs}")
        if shuffle and epoch > 0:
            x_batches, y_batches = shuffle_data(x, y, batch_size)

        for ix, (xb, yb) in enumerate(zip(x_batches, y_batches)):
            loss_value, acc = train_step(model, xb, yb, loss_obj, batch_size)
            tmp_acc += acc
            tmp_loss += loss_value
            print(f"[{ix+1} / {len(x_batches)}] acc: {acc} - loss: {loss_value}", end = '\r')
        
        for xv, yv in zip(x_val_batches, y_val_batches):
            val_loss, val_acc = val_step(model, xv, yv, loss_obj, batch_size)
            tmp_val_acc += val_acc
            tmp_val_loss += val_loss
                
        history['acc'].append(tmp_acc / len(x_batches))
        history['val_acc'].append(tmp_val_acc / len(x_val_batches))
        history['loss'].append(tmp_loss / len(x_batches))
        history['val_loss'].append(tmp_val_loss / len(x_val_batches))

        print(f"Training Loss: {loss_value}, Training Accuracy: {history['acc'][-1]}, Validation Loss: {val_loss}, Validation Accuracy: {history['val_acc'][-1]}")
    return history
     




if __name__ == "__main__":
    with mirrored_strategy.scope():

        print(f"{ts()} I source/model/train.py] Starting training...")
        ffolder_ckpt = Path(CKPT_FOLDER) / f"batch_size_{BATCH_SIZE}" / f"LambdaL1_{LMDA_L1}"
        print(f"{ts()} I source/model/train.py] Built checkpoint folders")

        try:
            os.makedirs(ffolder_ckpt)
        except:
            pass
        
        fpath_ckpt =  ffolder_ckpt / "cp.ckpt"
        print(f"{ts()} I source/model/train.py] Saving to {fpath_ckpt}")
        
        # ==================================================================
        #                  INIT Classes and Parameters
        # ------------------------------------------------------------------
        n = Network()
        n.construct(LMBDA_KERNEL, LMBDA_BIAS, LMBDA_ACTIVITY)
        cifar = CifarData()
        ckpt = tf.keras.callbacks.ModelCheckpoint(
                filepath = fpath_ckpt, monitor = 'val_loss', mode = 'min',
                verbose = 1, save_best_only = True, save_weights_only = False
        )
        # ==================================================================

        print(f"{ts()} I source/model/train.py] Loading Dataset ...")

        # ==================================================================
        #                  Load The Dataset
        # ------------------------------------------------------------------   
        train_data , val_data = cifar.load_data(size = 1000)   
        x_train, y_train, x_val, y_val = n.shape_td(train_data, val_data)
        # ==================================================================
        
        print(f"{ts()} I source/model/train.py] Dataset Loaded.")
        print(f"{ts()} I soruce/model/train.py] Fitting Model ...")

        # ==================================================================
        #                  Hyperparam Tuning
        # ------------------------------------------------------------------ 

        h = n.model.fit(x_train, y_train, batch_size = BATCH_SIZE,
            epochs = EPOCHS, validation_data = (x_val, y_val),
            callbacks = [ckpt])

        """
        h = fit(n, x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                x_val = x_val, y_val = y_val, shuffle=True)
        """
        # ================================================================== 

        print(f"{ts()} I source/model/train.py] Hyperparameter tuned. Plotting results...")
            
        # ==================================================================
        #                  Plotting
        # ------------------------------------------------------------------ 
        plt.plot(h['acc'])
        plt.plot(h['val_acc'])
        plt.legend(['training', 'validation'])
        plt.title(f"Training Acc for {EPOCHS} Epochs with {BATCH_SIZE} batch size.")
        plt.savefig(f"{ts_file()}_acc.png")

        plt.plot(h['loss'])
        plt.plot(h['val_loss'])
        plt.legend(['training', 'validation'])
        plt.title(f"Training Loss for {EPOCHS} Epochs with {BATCH_SIZE} batch size.")
        plt.savefig(f"{ts_file()}_loss.png")
        # ==================================================================

        n.model.save(f"{ts_file()}_model.h5")
        print("Finished train.py. Press any Key to exit...")
        wait = input()
