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


class generator(keras.utils.Sequence):

    def __init__(self, batch_size, validation=False):
        util.kaggle_and_extract("./dataset/FRUITS", "moltean/fruits")
        file_path = './dataset/FRUITS/fruit-360/Training' if not validation else './dataset/FRUITS/fruit-360/Test'
        keys = [
            str(x) for x in sorted(
                Path(file_path).glob('*'))
        ]
        self.x = list(
            sorted(
                Path(file_path).glob("**/*.jpg")))
        self.y = list(
            [keys.index('/'.join(str(x).split('/')[:-1])) for x in self.x])
        self.batch_size = batch_size
        self.size = len(self.x)
        self.inds = np.random.shuffle(np.arange(len(self.x)))

    def __len__(self):
        print("B")
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def on_epoch_end(self):
        self.inds = np.random.shuffle(np.arange(len(self.x)))

    def __getitem__(self, idx):
        print("A")
        batch = self.inds[idx * self.batch_size: (idx+1) * self.batch_size]
        # batch = np.random.choice(self.inds, self.batch_size, False)
        # self.inds = [x for x in self.inds if x not in batch]
        batch_x, batch_y = np.asarray([
            plt.imread(self.x[i]).astype(np.float64) / 255.0 for i in batch
        ]), np.asarray([self.y[i] for i in batch])
        print(batch_x, batch_y)
        return batch_x, batch_y
