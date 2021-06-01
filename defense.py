""" Combined Network Class for Defenses """
from numpy.core.numeric import indices
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import sys
from tensorflow.python.ops.gen_array_ops import zeros_like
from tqdm import tqdm
from tensorflow.core.framework.versions_pb2 import DESCRIPTOR

DROPOUT_TYPE = type(tf.keras.layers.core.Dropout(.1))
LAYER_INPUT_TYPE = type(tf.keras.layers.Input(shape = (32,32)))

class DNet:
    def __init__(self,):
        self.__name__ = "NoobNet"
        self.model = None
        self.optimizer = None
        self.loss_fun = None
        self.built = False

    def __call__(self, x, defense = None):
        if defense == "DD":
            return self.defensive_dropout(x)
        elif defense == "SAP":
            return self.stochastic_pruning(x, r = 0.9)
        else:
            return self.model(x)

    @tf.function
    def defensive_dropout(self, x, defense = True):
        """ 
            Runs inference on the nerual network.

            We do this by specifically by tricking the batch normalization
            layer into thinking that the network is being run in training
            mode. This causes the batchNorm layer to normalize the outputs.
            All other layers still perform inference as they normally would.

            Args:
                :param np.ndarray x: The input data
                :param bool defense: Whether or not to use defensive dropout
            
            Returns:
                :np.ndarray preds: the output of the network inference
        """
        if defense:
            for ix, layer in enumerate(self.model.layers):
                if ix == 0:
                    preds = layer(x)
                elif type(layer) == DROPOUT_TYPE:
                    preds = layer(preds, training = True)
                else:
                    preds = layer(preds)
            return preds
        else:
            return self.model(x)

    @tf.function
    def stochastic_pruning(self, x, r):
        """
            Runs the stochastic activation prunning as defined by 
            (Dhillon et al., 2018).

            Args:
                :param x: Input data to the first layer
                :param r: percent of activtions to keep
        """
        assert r <= 1
        h = x
        power = tf.constant(r, dtype=tf.float32)
        for layer in self.model.layers:
            if type(layer) != LAYER_INPUT_TYPE:
                samples = []
                print(type(layer))
                h = layer(h)
                h_flat = tf.reshape(h, [-1, 1])
                h_sum = tf.math.reduce_sum(h_flat)
                p_flat = tf.reshape(tf.divide(tf.math.abs(h_flat), h_sum), [-1, 1])
                """
                #print(p_flat)
                dist = tfp.distributions.Categorical(probs = p_flat)
                samp = tfp.distributions.Sample(dist, sample_shape = len(h))
                samples = samp.sample()
                new_h = tf.gather(h_flat, samples)
                """
                samples = tf.zeros_like(h_flat)
                for jx in range(int(r*len(h_flat))):
                    print(f"{jx} / {int(r*len(h_flat))}", end = '\r')
                    samples = tf.add(tf.cast(tf.random.categorical(p_flat, 1), tf.float32), samples)

                a_one = tf.multiply(tf.ones_like(h_flat), tf.cast(samples, tf.float32))
                samples = tf.minimum(samples, tf.ones_like(samples))
                hij = tf.multiply(h_flat, tf.cast(samples, tf.float32))
                pij = tf.multiply(p_flat, tf.cast(samples, tf.float32))
                new_h = tf.divide(hij, tf.subtract(a_one, tf.pow(tf.subtract(a_one, pij), power)))

                """
                for ix, s in enumerate(samples):
                    print(f"{ix}/ {len(samples)}", end='\r')
                    a_one = tf.multiply(tf.ones_like(h_flat, dtype = tf.float32), tf.cast(s, tf.float32))
                    hij = tf.multiply(h_flat, tf.cast(s, tf.float32))
                    pij = tf.multiply(p_flat, tf.cast(s, tf.float32))
                    fac = tf.divide(hij, tf.subtract(a_one, tf.pow(tf.subtract(a_one, pij), power)))
                    new_h = tf.add(new_h, fac)
                """
                #new_h = tf.multiply(tf.cast(samples, tf.float32), h_flat)
                #h = tf.reshape(new_h, tf.shape(h))
                
                h = tf.reshape(new_h, tf.shape(h))
                
        return h

    def shape_td(self, train_data, val_data):
        """ Formats data dictionaries to return x, y values for the NN.

        Args:
            :param dict train_data: Dictionary of train data, keys = "x_train", "y_train"
            :param dict val_data: Dictionary of validation data, keys = "x_val", "y_val"
        """
        x_train = np.array(train_data['x_train']).reshape(-1, 32, 32, 3)
        x_val = np.array(val_data['x_val']).reshape(-1, 32, 32, 3)
        y_train = np.array([tf.one_hot(y, 20) for y in train_data['y_train']])
        y_val = np.array([tf.one_hot(y, 20) for y in val_data['y_val']])
        return x_train, y_train, x_val, y_val
    
    def load_model(self, fpath, weights_only = False):
        """ 
        Loads a pre-trained neural model.

        Args:
            :param str fpath: The path to the .h5 model
            :param bool weights_only: Set to true if you are not loading a full network
        """
        if not weights_only:
            self.model = tf.keras.models.load_model(fpath)
        else:
            if not self.built:
                self.construct()
            self.model.load_weights(fpath)

    def construct(self, lmda_kernel = None, lmda_bias = None, lmda_activiy = None,
        optimizer = tf.keras.optimizers.Adam(learning_rate = 3e-4), loss_fn = tf.keras.losses.CategoricalCrossentropy()):
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
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(inpt)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 5, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 5, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
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
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same", 
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(drpt1)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy, activation = 'relu')(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
        conv = tf.keras.layers.Conv2D(16, 3, (1, 1), padding = "same",
            kernel_regularizer = lmda_kernel, bias_regularizer = lmda_bias,
            activity_regularizer = lmda_activiy)(bnorm)
        bnorm = tf.keras.layers.BatchNormalization()(conv)
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
        dense = tf.keras.layers.Dense(60, kernel_regularizer = lmda_kernel, 
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy)(fltn2)
        prelu = tf.keras.layers.PReLU()(dense)
        bnorm = tf.keras.layers.BatchNormalization()(prelu)
        drpt = tf.keras.layers.Dropout(0.05)(bnorm)
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # Dense Layer 
        #
        dense = tf.keras.layers.Dense(60, kernel_regularizer = lmda_kernel, 
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy)(drpt)
        prelu = tf.keras.layers.PReLU()(dense)
        bnorm = tf.keras.layers.BatchNormalization()(prelu)
        drpt = tf.keras.layers.Dropout(0.05)(bnorm)
        # -------------------------------------------------------------------

        # -------------------------------------------------------------------
        # Dense Layer 
        #
        dense = tf.keras.layers.Dense(60, kernel_regularizer = lmda_kernel, 
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy)(drpt)
        prelu = tf.keras.layers.PReLU()(dense)
        bnorm = tf.keras.layers.BatchNormalization()(prelu)
        drpt = tf.keras.layers.Dropout(0.05)(bnorm)
        # -------------------------------------------------------------------
        
        # -------------------------------------------------------------------
        # Dense Layer 
        #
        dense = tf.keras.layers.Dense(60, kernel_regularizer = lmda_kernel,\
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy)(drpt)
        prelu = tf.keras.layers.PReLU()(dense)
        bnorm = tf.keras.layers.BatchNormalization()(prelu)
        # -------------------------------------------------------------------
        
        # -------------------------------------------------------------------
        # Output Layer 
        #
        otpt = tf.keras.layers.Dense(20, kernel_regularizer = lmda_kernel, 
            bias_regularizer = lmda_bias, activity_regularizer = lmda_activiy, 
            activation = 'softmax')(bnorm)
        # -------------------------------------------------------------------


        self.model = tf.keras.Model(inputs = inpt, outputs = otpt)
        self.loss_fun = loss_fn
        self.optimizer = optimizer
        """
        self.model.compile(
            loss = tf.keras.losses.CategoricalCrossentropy(),
            optimizer = tf.keras.optimizers.Adam(lr = 3e-4),
            metrics = ['accuracy'],
        )
        """
        #self.model.summary()
        self.built = True
        try:
            tf.keras.utils.plot_model(self.model, "model.png")
        except:
            raise Warning("You have not installed graphviz so the model architecutre will not be plotted to a .png file!")