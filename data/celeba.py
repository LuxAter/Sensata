#!/usr/bin/env python3
import csv
import numpy as np
import data.util as util
from pathlib import Path
import pprint


def load_train_data():
    util.kaggle_and_extract_all("./dataset/CELEBA",
            "jessicali9530/celeba-dataset")
    csv_data = list(csv.reader(open('./dataset/CELEBA/list_attr_celeba.csv')))
    keys_train = csv_data[0][1:]
    np.random.seed(2019)
    Y_all = [(np.asarray(x[1:], dtype=np.float64)+1.0)/2.0 for x in csv_data[1:]]
    X_all = ["./dataset/CELEBA/img_align_celeba/{}".format(x[0]) for x in csv_data[1:]]
    train_inds = np.random.choice(range(len(Y_all)), len(Y_all) * 0.9)
    return [x for i, x in enumerate(X_all) if i in train_inds], [x for i, x in enumerate(Y_all) if i in train_inds]


def load_test_data():
    util.kaggle_and_extract_all("./dataset/CELEBA",
            "jessicali9530/celeba-dataset")
    csv_data = list(csv.reader(open('./dataset/CELEBA/list_attr_celeba.csv')))
    keys_test = csv_data[0][1:]
    np.random.seed(2019)
    Y_all = [(np.asarray(x[1:], dtype=np.float64)+1.0)/2.0 for x in csv_data[1:]]
    X_all = ["./dataset/CELEBA/img_align_celeba/{}".format(x[0]) for x in csv_data[1:]]
    train_inds = np.random.choice(range(len(Y_all)), int(np.floor(len(Y_all) * 0.9)))
    return [x for i, x in enumerate(X_all) if i not in train_inds], [x for i, x in enumerate(Y_all) if i not in train_inds]


def load_data():
    return (load_train_data(), load_test_data())


def load_class_name():
    return list(csv.reader(open('./dataset/CELEBA/list_attr_celeba.csv',
        'r')))[0][1:]


    def generator(batch_size, validation=False):
        util.kaggle_and_extract_all("./dataset/CELEBA",
                "jessicali9530/celeba-dataset")
        csv_data = list(csv.reader(open('./dataset/CELEBA/list_attr_celeba.csv')))
    keys_train = csv_data[0][1:]
    np.random.seed(2019)
    Y_all = [(np.asarray(x[1:], dtype=np.float64)+1.0)/2.0 for x in csv_data[1:]]
    X_all = ["./dataset/CELEBA/img_align_celeba/{}".format(x[0]) for x in csv_data[1:]]
    train_inds = np.random.choice(range(len(Y_all)), len(Y_all) * 0.9)
    if validation:
        return util.Generator([x for i, x in enumerate(X_all) if i not in train_inds],
                [y for i, y in enumerate(Y_all) if i not in train_inds],
                batch_size=batch_size,
                dim=(218, 178, 3),
                n_classes=len(keys_train))
    else:
        return util.Generator([x for i, x in enumerate(X_all) if i in train_inds],
                [y for i, y in enumerate(Y_all) if i in train_inds],
                batch_size=batch_size,
                dim=(218, 178, 3),
                n_classes=len(keys_train))
