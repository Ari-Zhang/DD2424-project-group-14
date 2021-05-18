""" Config file for the Neural Network CNN architecture. """
import tensorflow as tf
import numpy as np
from tensorflow.core.framework.versions_pb2 import DESCRIPTOR

class Network:
    def __init__(self,):
        self.__name__ = "NoobNet"
        self.model = None
    
    def shape_td(self, train_data, val_data):
        """ Formats data dictionaries to return x, y values for the NN.

        Args:
            :param dict train_data: Dictionary of train data, keys = "x_train", "y_train"
            :param dict val_data: Dictionary of validation data, keys = "x_val", "y_val"
        """
        x_train = np.array(train_data['x_train']).reshape(-1, 32, 32, 3)
        x_val = np.array(val_data['x_val']).reshape(-1, 32, 32, 3)
        y_train = np.array([tf.one_hot(y, 100) for y in train_data['y_train']])
        y_val = np.array([tf.one_hot(y, 100) for y in val_data['y_val']])
        return x_train, y_train, x_val, y_val
    
    def construct(self, lmda_kernel = None, lmda_bias = None, lmda_activiy = None):
        """ 
        Builds and Compiles the classifier.

        Use this function to quickly prepare the model for being used together
        with the tf.model.fit function. Mainly we use this to search and quickly 
        iterate over many hyperparameters, so every time you rebuild with different
        params you can just recompile the model using this function.

        Args:
            :param float32 lmda_kernel: The gegulization term applied to all weights
            :param float32 lmda_bias: The regulization term applied to all bias vectors
            :param float32 lmda_activity: The regularization term applied to the entire layer
        """

        inpt = tf.keras.layers.Input(shape = (32, 32, 3))
        
        # -------------------------------------------------------------------
        # Convolutional Layer 
        #
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(inpt)
        conv = tf.keras.layers.Conv2D(32, 5, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(32, 5, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy)(conv)
        prelu = tf.keras.layers.PReLU()(conv)
        pool = tf.keras.layers.MaxPool2D(2,2)(prelu)
        bnorm = tf.keras.layers.BatchNormalization()(pool)
        drpt1 = tf.keras.layers.Dropout(0.075)(bnorm)
        # --------------------------------------------------------------------

        # -------------------------------------------------------------------
        # Convolutional Layer 
        #
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same", 
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(drpt1)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(conv)
        conv = tf.keras.layers.Conv2D(64, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy)(conv)
        prelu = tf.keras.layers.PReLU()(conv)
        pool = tf.keras.layers.MaxPool2D(2,2)(prelu)
        bnorm = tf.keras.layers.BatchNormalization()(pool)
        drpt2 = tf.keras.layers.Dropout(0.075)(bnorm)
        # --------------------------------------------------------------------
        
        # Flatten
        fltn2 = tf.keras.layers.Flatten()(drpt2)

        # -------------------------------------------------------------------
        # Dense Layer 
        #
        dense = tf.keras.layers.Dense(50, kernel_regularizer = lmda_kernel, 
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy)(fltn2)
        prelu = tf.keras.layers.PReLU()(dense)
        drpt = tf.keras.layers.Dropout(0.05)(prelu)
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # Dense Layer 
        #
        dense = tf.keras.layers.Dense(300, kernel_regularizer = lmda_kernel, 
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy)(drpt)
        prelu = tf.keras.layers.PReLU()(dense)
        drpt = tf.keras.layers.Dropout(0.05)(prelu)
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # Dense Layer 
        #
        dense = tf.keras.layers.Dense(300, kernel_regularizer = lmda_kernel, 
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy)(drpt)
        prelu = tf.keras.layers.PReLU()(dense)
        drpt = tf.keras.layers.Dropout(0.05)(prelu)
        # -------------------------------------------------------------------
        
        # -------------------------------------------------------------------
        # Dense Layer 
        #
        dense = tf.keras.layers.Dense(500, kernel_regularizer = lmda_kernel,\
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy)(drpt)
        prelu = tf.keras.layers.PReLU()(dense)
        # -------------------------------------------------------------------
        
        # -------------------------------------------------------------------
        # Output Layer 
        #
        otpt = tf.keras.layers.Dense(100, kernel_regularizer = lmda_kernel, 
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy, 
            activation = 'softmax')(prelu)
        # -------------------------------------------------------------------


        self.model = tf.keras.Model(inputs = inpt, outputs = otpt)
        self.model.compile(
            loss = tf.keras.losses.CategoricalCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(lr = 3e-4),
            metrics = ['accuracy'],
        )
        self.model.summary()
        tf.keras.utils.plot_model(self.model, "model.png")



        
        

