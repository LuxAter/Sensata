#!/usr/bin/env python3
import csv
import numpy as np
import data.util as util
from pathlib import Path
import pprint

def load_train_data():
    util.kaggle_and_extract_all("./dataset/CELEBA", "jessicali9530/celeba-dataset")

def load_test_data():
    util.kaggle_and_extract_all("./dataset/CELEBA", "jessicali9530/celeba-dataset")

def load_data():
    return (load_train_data(), load_test_data())

def load_class_name():
    return list(csv.reader(open('./dataset/CELEBA/list_attr_celeba.csv', 'r')))[0][1:]

def generator(batch_size, validation=False):
    util.kaggle_and_extract_all("./dataset/CELEBA", "jessicali9530/celeba-dataset")
    X_train = sorted(Path('./dataset/CELEBA/img_align_celeba').glob("*.jpg"))
    csv_data = list(csv.reader(open('./dataset/CELEBA/list_attr_celeba.csv')))
    keys_train = csv_data[0][1:]
    Y_train = [np.asarray(x[1:]) for x in csv_data[1:]]
    return util.Generator(X_train, Y_train, batch_size=batch_size, dim=(218, 178, 3), n_classes=len(keys_train))
