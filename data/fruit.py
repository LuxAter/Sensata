#!/usr/bin/env python3
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import data.util as util
from tensorflow import keras


def load_train_data():
    util.kaggle_and_extract("./dataset/FRUITS", "moltean/fruits")
    keys = [
        str(x)
        for x in sorted(Path('./dataset/FRUITS/fruits-360/Training').glob('*'))
    ]
    X_train = np.asarray(
        sorted(Path('./dataset/FRUITS/fruits-360/Training').glob("**/*.jpg")))
    Y_train = np.asarray(
        [keys.index('/'.join(str(x).split('/')[:-1])) for x in X_train])
    return X_train, Y_train, util.class_encoding(Y_train)


def load_test_data():
    util.kaggle_and_extract("./dataset/FRUITS", "moltean/fruits")
    keys = [
        str(x)
        for x in sorted(Path('./dataset/FRUITS/fruits-360/Test').glob('*'))
    ]
    X_test = np.asarray(
        sorted(Path('./dataset/FRUITS/fruits-360/Test').glob("**/*.jpg")))
    Y_test = np.asarray(
        [keys.index('/'.join(str(x).split('/')[:-1])) for x in X_test])
    return X_test, Y_test, util.class_encoding(Y_test)


def load_data():
    return (load_train_data(), load_test_data())


def load_class_name():
    return [
        str(x).split('/')[-1]
        for x in sorted(Path('./dataset/FRUITS/fruits-360/Test').glob('*'))
    ]


def generator(batch_size, validation=False):
    util.kaggle_and_extract("./dataset/FRUITS", "moltean/fruits")
    if validation:
        keys_test = [
            str(x)
            for x in sorted(Path('./dataset/FRUITS/fruits-360/Test').glob('*'))
        ]
        X_test = sorted(
            Path('./dataset/FRUITS/fruits-360/Test').glob("**/*.jpg"))
        Y_test = [
            keys_test.index('/'.join(str(x).split('/')[:-1])) for x in X_test
        ]
        return util.Generator(X_test,
                              Y_test,
                              batch_size=batch_size,
                              dim=(100, 100, 3),
                              n_classes=len(keys_test))
    else:
        keys_train = [
            str(x) for x in sorted(
                Path('./dataset/FRUITS/fruits-360/Training').glob('*'))
        ]
        X_train = sorted(
            Path('./dataset/FRUITS/fruits-360/Training').glob("**/*.jpg"))
        Y_train = [
            keys_train.index('/'.join(str(x).split('/')[:-1])) for x in X_train
        ]
        return util.Generator(X_train,
                              Y_train,
                              batch_size=batch_size,
                              dim=(100, 100, 3),
                              n_classes=len(keys_train))
