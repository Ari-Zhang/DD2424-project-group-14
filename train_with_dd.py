""" Extra training functions for defensive dropout

These functions allow you to train the network with 
defensive dropout enabled already in the validation
steps. Could be useful i guess.

"""
from defensive_dropout import DDNet
from data import CifarData
import matplotlib.pyplot as plt
from datetime import datetime as dt
import numpy as np
import tensorflow as tf
import os
from pathlib import Path

DATA_PATH = "D:\\KTH\\courses\\dd2424\\projects\\data\\cifar-100-python"
CKPT_FOLDER = "D:\\KTH\\courses\\dd2424\\projects\\model\\ckpts"
BATCH_SIZE = 300
EPOCHS = 15
LMDA_L1 = .005
LMDA_L2 = .001
LMBDA_ACTIVITY = None #tf.keras.regularizers.L1(l1 = LMDA_L1)
LMBDA_KERNEL = None #tf.keras.regularizers.L1L2(l1 = LMDA_L1, l2 = LMDA_L2)
LMBDA_BIAS = None #tf.keras.regularizers.L1L2(l1 = LMDA_L1, l2 = LMDA_L2)

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
def train_step(model, x, y):
    with tf.GradientTape() as tape:
        logits = model(x, defense = True) # run model with defensive dropout
        loss_value = model.loss_fun(y, logits)
        grads = tape.gradient(loss_value, model.model.trainable_weights)
        model.optimizer.apply_gradients(zip(grads, model.model.trainable_weights))
        acc = accuracy(y, logits)
    return loss_value, acc

@tf.function
def val_step(model, x, y):
    val_logits = model(x, defense = True)
    val_loss = model.loss_fun(y, val_logits)
    val_acc = accuracy(y, val_logits)
    return val_loss, val_acc

@tf.function
def float_round(x, decimals = 0):
    multiplier = tf.constant(10**decimals, dtype=x.dtype)
    return tf.round(x * multiplier) / multiplier

def fit(model, x, y, x_val, y_val, epochs = EPOCHS, batch_size = BATCH_SIZE):
    """ 
    Fits the network to the dataset 

    Args:
        :param tf.keras.model model: A tensorflow model
        :param iterable x: The training datas input
        :param iterable y: The training data truths
        :param iterable x_val: The validation data input
        :param iterable y_val: The validation data truths
        :param int epochs: Number of training epochs
        :param int batch_size: The number of samples in each batch
    Returns:
        :dict history: Training and Validation Losses and Accuracy
    """
    x_batches = np.array_split(x, int(len(x) / batch_size))
    y_batches = np.array_split(y, int(len(x) / batch_size))
    x_val_batches = np.array_split(x_val, int(len(x_val) / batch_size))
    y_val_batches = np.array_split(y_val, int(len(x_val) / batch_size))
    history = {'accuracy': [], 'val_accuracy': []}
    
    for epoch in range(epochs):
        tmp_acc = 0
        tmp_val_acc = 0
        print(f"\nEpoch: {epoch+1} out of {epochs}")
        for ix, (xb, yb) in enumerate(zip(x_batches, y_batches)):
            loss_value, acc = train_step(model, xb, yb)
            tmp_acc += acc

            print(f"[Batch {ix + 1} / {len(x_batches)}] Loss: {loss_value} Accuracy: {tmp_acc / ix}, {acc}", end = "\r")
        
        for xv, yv in zip(x_val_batches, y_val_batches):
            val_loss, val_acc = val_step(model, xv, yv)
            tmp_val_acc += val_acc
            
        
        history['accuracy'].append(tmp_acc / len(x_batches))
        history['val_accuracy'].append(tmp_val_acc / len(x_batches))
        print(f"Validation Loss: {val_loss}, Validation Accuracy: {history['val_accuracy'][-1]}")
    
    return history

def main():
    cifar = CifarData(0, DATA_PATH)
    model = DDNet()
    #model.construct()
    model.load_model("final.h5")
    model.loss_fun = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
    model.optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4)
    train_data , val_data = cifar.load_data(size = 6000)   
    x_train, y_train, x_val, y_val = model.shape_td(train_data, val_data)
    h = fit(model, x_train, y_train, x_val, y_val, 6, 300)

    plt.plot(h['accuracy'])
    plt.plot(h['val_accuracy'])
    plt.title("Validation using Defensive Dropout")
    plt.xlabel('training steps')
    plt.ylabel('Accuracy')
    plt.legend(['acc', 'val_acc'])
    plt.show()
    return 0

if __name__ == "__main__":
    result = main()