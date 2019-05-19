#!/usr/bin/env python3
import data.util as util
import numpy as np
from tensorflow import keras

data_shape = (32, 32, 3)


def load_train_data():
    util.download_and_extract(
        "./dataset/CIFAR-10",
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    X_train = np.zeros(shape=[50000, 32, 32, 3], dtype=float)
    Y_train = np.zeros(shape=[50000], dtype=int)
    begin = 0
    for i in range(1, 6):
        data_dict = util.unpickle(
            "./dataset/CIFAR-10/cifar-10-batches-py/data_batch_{}".format(i))
        x_raw = (np.array(data_dict[b'data'], dtype=float) / 255.0).reshape(
            [-1, 3, 32, 32]).transpose([0, 2, 3, 1])
        y_raw = np.array(data_dict[b'labels'], dtype=int)
        num = len(x_raw)
        end = begin + num
        X_train[begin:end, :] = x_raw
        Y_train[begin:end] = y_raw
        begin = end
    return X_train, Y_train, util.class_encoding(Y_train)


def load_test_data():
    util.download_and_extract(
        "./dataset/CIFAR-10",
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz")
    data_dict = util.unpickle("./dataset/CIFAR-10/cifar-10-batches-py/test_batch")
    X_test = (np.array(data_dict[b'data'], dtype=float) / 255.0).reshape(
        [-1, 3, 32, 32]).transpose([0, 2, 3, 1])
    Y_test = np.array(data_dict[b'labels'], dtype=int)
    return X_test, Y_test, util.class_encoding(Y_test)


def load_data():
    return (load_train_data(), load_test_data())


def load_class_name():
    raw = util.unpickle(
        "./dataset/CIFAR-10/cifar-10-batches-py/batches.meta")[b'label_names']
    return [x.decode('utf-8') for x in raw]

