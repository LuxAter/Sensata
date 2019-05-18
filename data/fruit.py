#!/usr/bin/env python3
import csv
import numpy as np
import data.util as util


def load_train_data():
    util.kaggle_and_extract("./dataset/FRUITS", "moltean/fruits")
    # raw = list(csv.reader(open('./dataset/fruits', 'r')))[1:]
    # X_train = np.asarray([np.asarray([x for x in r[:-1]]) for r in raw])
    # Y_train = np.asarray([np.asarray(get_key(r[-1])) for r in raw])
    # return X_train, Y_train, util.class_encoding(Y_train)


def load_test_data():
    pass
    # util.kaggle_and_extract("./dataset/IRIS", "uciml/iris")
    # raw = list(csv.reader(open('./dataset/IRIS/Iris.csv', 'r')))[1:]
    # X_test = np.asarray([np.asarray([x for x in r[:-1]]) for r in raw])
    # Y_test = np.asarray([np.asarray(get_key(r[-1])) for r in raw])
    # return X_test, Y_test, util.class_encoding(Y_test)


def load_data():
    return (load_train_data(), load_test_data())


def load_class_name():
    return ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]

class generator(keras.utils.Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.on_epoch_end]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

