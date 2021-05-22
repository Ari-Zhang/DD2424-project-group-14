from data import CifarData
from sa_pruning import SAPNet
import tensorflow as tf
import numpy as np

def main():
    print(tf.executing_eagerly())
    cifar = CifarData()
    d = cifar.load_data(size = 100)
    net = SAPNet()
    net.load_model("final.h5")
    net.model.summary()
    x, y, xv, yv = net.shape_td(d[0], d[1])
    preds = net.stochastic_pruning(x, 0.9)
    print(preds.shape)
    print(np.argmax(preds, axis = 1))
    print(np.max(preds, axis = 1))

if __name__ == "__main__":
    main()