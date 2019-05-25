#!/usr/bin/env python3

from time import time

import tensorflow as tf
from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

import data.mnist


def main():
    (train_images, train_labels, _), (test_images, test_labels,
                                      _) = data.mnist.load_data()
    # names = data.mnist.load_class_name()
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    file_path = ".{}_log/{}".format(__file__.strip("./").strip(".py"), time())
    tensorboard = keras.callbacks.TensorBoard(log_dir=file_path)
    data_save = keras.callbacks.CSVLogger('{}/log.csv'.format(file_path),
                                          append=True,
                                          separator=',')
    model_save = keras.callbacks.ModelCheckpoint(
        '{}/{{epoch:05}}.h5'.format(file_path), period=10)
    model.summary()
    model.fit(train_images,
              train_labels,
              epochs=100,
              callbacks=[tensorboard, model_save, data_save],
              validation_data=(test_images, test_labels))


if __name__ == "__main__":
    main()
