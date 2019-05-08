#!/usr/bin/env python3
import data.util as util
import numpy as np

data_shape = (28, 28)


def load_train_data():
    util.download_and_extract(
        "./dataset/MNIST", "http://deeplearning.net/data/mnist/mnist.pkl.gz")
    train, valid, test = util.unpickle("./dataset/MNIST/mnist.pkl")
    X_train = np.concatenate([train[0], valid[0]]).reshape([-1, 28, 28])
    Y_train = np.concatenate([train[1], valid[1]]).T
    return X_train, Y_train, util.class_encoding(Y_train)


def load_test_data():
    util.download_and_extract(
        "./dataset/MNIST", "http://deeplearning.net/data/mnist/mnist.pkl.gz")
    train, valid, test = util.unpickle("./dataset/MNIST/mnist.pkl")
    X_test = test[0].reshape([-1, 28, 28])
    Y_test = test[1].T
    return X_test, Y_test, util.class_encoding(Y_test)


def load_data():
    return (load_train_data(), load_test_data())


def load_class_name():
    return [str(x) for x in range(0, 10)]
