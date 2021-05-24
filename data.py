""" Data Loading Module for DD2424 Project.

This script loads, pre-processes and augments the cifar-100 dataset in such 
a way that we can train a tensorflow model with it."""
import numpy as np
import concurrent.futures
import multiprocessing as mp
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from itertools import chain

CPU_LIMIT = mp.cpu_count() - 1 # reserve 1 core for the OS
DATA_PATH = "D:\\KTH\\courses\\dd2424\\projects\\data\\cifar-100-python"
R = 4

class CifarData:
    """ 
    Class that controls the data loaded for our neural network. 

    We use the CIFAR-100 dataset in our tests, this class provides an interface
    to the CIFAR datasets as well as the ability to load, augment, and stream
    data batches on demand.
    """
    def __init__(self, fpath = DATA_PATH, mode = 'all'):
        """ 
        Initialized the CifarData Class.

        Args:
            :param str fpath: The path to train andtest data
            :param str mode: The data streaming mode ('all' or 'stream') use this to adjust for RAM shortage
        """
        self.fpath = fpath
        self.source_data = None
        self.thread_data = None
        self.stream = True if mode == 'stream' else False
    
    def load_data(self, size = None):
        """ Streams a data block back to the caller."""
        if self.stream:
            return self.__data_stream()
        else:
            return (self.__data_load_all(size), self.__data_load_val())

    def __data_load_all(self, size):
        """ Returns all data from the input folder."""
        ftrain = Path(self.fpath) / "train"
        train_p = open(ftrain, 'rb')
        train = pickle.load(train_p, encoding = "bytes")
        data = {"x_train": train[b'data'], "y_train": train[b'fine_labels']}
        data['x_train'] = data['x_train'].reshape((-1, 3, 32, 32))
        data['x_train'] = data['x_train'].T.astype(float)
        data['x_train'] = np.moveaxis(data['x_train'], -1, 0) # we need the last axis to be first to iterate over the data array
        data = self.__normalize(data)
        self.source_data = data
        train_data = {'x_train': [], 'y_train': []}
        aug_data = self.augment(size)
        for item in aug_data:
            train_data['x_train'].extend(item['x_train'])
            train_data['y_train'].extend(item['y_train'])
        train_data = self.shuffle(train_data)
        return train_data

    def __data_load_val(self, ):
        fval = Path(self.fpath) / "test"
        test_p = open(fval, 'rb')
        test = pickle.load(test_p, encoding = "bytes")
        data = {"x_val": test[b'data'], "y_val": test[b'fine_labels']}
        data['x_val'] = data['x_val'].reshape((-1, 3, 32, 32))
        data['x_val'] = data['x_val'].T.astype(float)
        data['x_val'] = np.moveaxis(data['x_val'], -1, 0) # we need the last axis to be first to iterate over the data array
        data = self.__normalize(data)
        return data
         
    def __normalize(self, data):
        """ Normalizes the training data x-values """
        keys = list(data.keys())
        for k in keys:
            if "x" in k:
                data[k] = (np.array(data[k]) - 128) / 128    
        return data
    
    def shuffle(self, data):
        idxes = np.arange(len(data[list(data.keys())[0]]))
        np.random.shuffle(idxes)
        keys = list(data.keys())
        for k in keys:
            data[k] = np.array(data[k])[idxes]
        return data
    
    def augment(self, size = None):
        """ 
        Extends the data set through augmentation. 

        We need to run these in separate iterations unfortunately due to
        memory errors that show up when running too many different ops 
        running on the same Spawn.        
        """
        if size is None:
            size = len(self.source_data['x_train'])

        r = (process_map(self._augment_thread_rotate,
                        range(size),
                        max_workers = CPU_LIMIT, 
                        chunksize = 3000))       
        r.extend(process_map(self._augment_thread_flip,
                        range(size),
                        max_workers = CPU_LIMIT, 
                        chunksize = 3000))
        return r
    
    def _augment_thread_flip(self, i):
        ret = {"x_train": [], "y_train": []}
        img = self.source_data['x_train'][i]
        fimg = np.fliplr(img)
        for j in range(R):
            rfimg = rotate(fimg, j*90).reshape(1, 32, 32, 3)
            ret['y_train'].append(self.source_data['y_train'][i])
            ret['x_train'].append(rfimg)
        return ret

    def _augment_thread_rotate(self, i):
        """ Threaded processing function. """
        ret = {"x_train": [], "y_train": []}
        img = self.source_data['x_train'][i]
        for j in range(R):
            rimg = rotate(img, j*90).reshape(1, 32, 32, 3)
            ret['y_train'].append(self.source_data['y_train'][i])
            ret['x_train'].append(rimg)
        return ret

