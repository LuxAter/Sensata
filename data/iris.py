#!/usr/bin/env python3
import csv
import numpy as np
import data.util as util


def get_key(key):
    if key == "Iris-setosa":
        return 0
    elif key == "Iris-versicolor":
        return 1
    elif key == "Iris-virginica":
        return 2


def load_train_data():
    util.kaggle_and_extract("./dataset/IRIS", "uciml/iris")
    raw = list(csv.reader(open('./dataset/IRIS/Iris.csv', 'r')))[1:]
    X_train = np.asarray([np.asarray([x for x in r[:-1]]) for r in raw])
    Y_train = np.asarray([np.asarray(get_key(r[-1])) for r in raw])
    return X_train, Y_train, util.class_encoding(Y_train)


def load_test_data():
    util.kaggle_and_extract("./dataset/IRIS", "uciml/iris")
    raw = list(csv.reader(open('./dataset/IRIS/Iris.csv', 'r')))[1:]
    X_test = np.asarray([np.asarray([x for x in r[:-1]]) for r in raw])
    Y_test = np.asarray([np.asarray(get_key(r[-1])) for r in raw])
    return X_test, Y_test, util.class_encoding(Y_test)


def load_data():
    return (load_train_data(), load_test_data())


def load_class_name():
    return ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
