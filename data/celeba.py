#!/usr/bin/env python3
import csv
import numpy as np
import data.util as util

def load_train_data():
    util.kaggle_and_extract("./dataset/CELEBA", "jessicali9530/celeba-dataset")
    # raw = list(csv.reader(open('./dataset/IRIS/Iris.csv', 'r')))[1:]
    # X_train = np.asarray([np.asarray([x for x in r[:-1]]) for r in raw])
    # Y_train = np.asarray([np.asarray(get_key(r[-1])) for r in raw])
    # return X_train, Y_train, util.class_encoding(Y_train)


def load_test_data():
    util.kaggle_and_extract("./dataset/IRIS", "uciml/iris")


def load_data():
    return (load_train_data(), load_test_data())


def load_class_name():
    return []
