""" Sequence for training and hyperparam tuning."""

from tensorflow.keras import callbacks, optimizers
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

strategy = tf.distribute.MirroredStrategy()

ap = argparse.ArgumentParser()
ap.add_argument("-epochs", "--epochs", type=int, required=True, help="how many epochs to train for")
ap.add_argument("-batchsize", "--batchsize", type=int, required=True, help="what batchsize to use")
ap.add_argument("-size", "--size", type=int, required=True, help="amount of data to train on")
ap.add_argument("-cpulim", "--cpulim", type=int, required=True, help="amount of data to train on")
ap.add_argument("-cnksize", "--cnksize", type=int, required=True, help="amount of data to train on")
args = vars(ap.parse_args())

SIZE = args['size']
CPU_LIMIT = args['cpulim']
CHUNK_SIZE = args['cnksize']
CKPT_FOLDER = "PLACEHOLDER"
BATCH_SIZE = args['batchsize']
GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
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
        loss = loss_obj(y, logits)
        loss_value = tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)
        grads = tape.gradient(loss_value, model.model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.model.trainable_weights))
        acc = accuracy(y, logits)
    return loss_value, acc

@tf.function
def val_step(model, x, y, loss_obj, batch_size):
    val_logits = model.model(x, training = False)
    loss = loss_obj(y, val_logits)
    val_loss = tf.nn.compute_average_loss(loss, global_batch_size=GLOBAL_BATCH_SIZE)
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
    
    loss_obj = tf.keras.losses.CategoricalCrossentropy(from_logits=True,
        reduction=tf.keras.losses.Reduction.NONE)

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
    with strategy.scope():

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
        n.construct(LMBDA_KERNEL, LMBDA_BIAS, LMBDA_ACTIVITY, optimizer = tf.keras.optimizers.Adam(lr=3e-4),
            loss_fn = None)
        cifar = CifarData(cnk = CHUNK_SIZE, cpulim = CPU_LIMIT)
        ckpt = tf.keras.callbacks.ModelCheckpoint(
                filepath = fpath_ckpt, monitor = 'val_loss', mode = 'min',
                verbose = 1, save_best_only = True, save_weights_only = False
        )
        # ==================================================================

        print(f"{ts()} I source/model/train.py] Loading Dataset ...")

        # ==================================================================
        #                  Load The Dataset
        # ------------------------------------------------------------------   
        train_data , val_data = cifar.load_data(size = SIZE)   
        x_train, y_train, x_val, y_val = n.shape_td(train_data, val_data)
        # ==================================================================
        
        print(f"{ts()} I source/model/train.py] Dataset Loaded.")
        print(f"{ts()} I soruce/model/train.py] Fitting Model ...")

        # ==================================================================
        #                  Hyperparam Tuning
        # ------------------------------------------------------------------ 
        
        n.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
            loss='categorical_crossentropy',
            metrics=['acc'])
        h = n.model.fit(x_train, y_train, batch_size = BATCH_SIZE,
            epochs = EPOCHS, validation_data = (x_val, y_val), shuffle = True,
            steps_per_epoch = int(len(x_train) / GLOBAL_BATCH_SIZE))
        h = h.history
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
        plt.clf()

        plt.plot(h['loss'])
        plt.plot(h['val_loss'])
        plt.legend(['training', 'validation'])
        plt.title(f"Training Loss for {EPOCHS} Epochs with {BATCH_SIZE} batch size.")
        plt.savefig(f"{ts_file()}_loss.png")
        # ==================================================================

        n.model.save(f"{ts_file()}_model.h5")
        print("Finished train.py. Press any Key to exit...")
        wait = input()
