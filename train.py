""" Sequence for training and hyperparam tuning."""

from tensorflow.keras import callbacks
from data import CifarData
from network import Network
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
from datetime import datetime as dt
import tensorflow as tf
import os
from pathlib import Path

DATA_PATH = "D:\\KTH\\courses\\dd2424\\projects\\data\\cifar-100-python"
CKPT_FOLDER = "D:\\KTH\\courses\\dd2424\\projects\\model\\ckpts"
BATCH_SIZE = 500
EPOCHS = 15
LMDA_L1 = .005
LMDA_L2 = .001
LMBDA_ACTIVITY = None #tf.keras.regularizers.L1(l1 = LMDA_L1)
LMBDA_KERNEL = None #tf.keras.regularizers.L1L2(l1 = LMDA_L1, l2 = LMDA_L2)
LMBDA_BIAS = None #tf.keras.regularizers.L1L2(l1 = LMDA_L1, l2 = LMDA_L2)

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
    ffolder_ckpt = Path(CKPT_FOLDER) / f"batch_size_{BATCH_SIZE}" / f"LambdaL2_{LMDA_L2}"
    print(f"{ts()} I source/model/train.py] Built checkpoint folders")

    
    
    fpath_ckpt =  ffolder_ckpt / "cp.ckpt"
    print(f"{ts()} I source/model/train.py] Saving to {fpath_ckpt}")
    
    # ==================================================================
    #                  INIT Classes and Parameters
    # ------------------------------------------------------------------
    n = Network()
    n.construct(LMBDA_KERNEL, LMBDA_BIAS, LMBDA_ACTIVITY)
    cifar = CifarData(8, DATA_PATH)
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
            epochs = 3, validation_data = (x_val, y_val),
            callbacks = [ckpt])
        plt.plot(h.history['val_accuracy'])
        lgnd.append(lm)
    # ==================================================================

    plt.title('Val Accuracy for Lambas for 3 epochs')
    plt.legend(lgnd)
    plt.show()
     




if __name__ == "__main__":
    coarse_search(0, 1, 0.5)
    """
    print(f"{ts()} I source/model/train.py] Starting training...")
    ffolder_ckpt = Path(CKPT_FOLDER) / f"batch_size_{BATCH_SIZE}" / f"LambdaL2_{LMDA_L2}"
    print(f"{ts()} I source/model/train.py] Built checkpoint folders")

    os.makedirs(ffolder_ckpt)
    
    fpath_ckpt =  ffolder_ckpt / "cp.ckpt"
    print(f"{ts()} I source/model/train.py] Saving to {fpath_ckpt}")
    
    # ==================================================================
    #                  INIT Classes and Parameters
    # ------------------------------------------------------------------
    n = Network()
    n.construct(LMBDA_KERNEL, LMBDA_BIAS, LMBDA_ACTIVITY)
    cifar = CifarData(8, DATA_PATH)
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
    h = n.model.fit(x_train, y_train, batch_size = BATCH_SIZE,
        epochs = EPOCHS, validation_data = (x_val, y_val),
        callbacks = [ckpt])
    # ================================================================== 

    print(f"{ts()} I source/model/train.py] Hyperparameter tuned. Plotting results...")
        
    # ==================================================================
    #                  Plotting
    # ------------------------------------------------------------------ 
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.legend(['training', 'validation'])
    plt.title(f"Training for {EPOCHS} Epochs with {BATCH_SIZE} batch size.")
    plt.show()
    # ==================================================================
    """
    n.model.save("final.h5")
    wait = input()
