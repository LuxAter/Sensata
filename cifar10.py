#!/usr/bin/env python3

from time import time

import tensorflow as tf
from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

import data.cifar10


def main():
    (train_images, train_labels, _), (test_images, test_labels,
                                      _) = data.cifar10.load_data()
    # names = data.cifar10.load_class_name()
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(32, 32, 3)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=".cifar10_log/{}".format(time()))
    model_save = keras.callbacks.ModelCheckpoint('.cifar10_log/{epoch:05}.h5',
                                                 period=10)
    model.fit(train_images,
              train_labels,
              epochs=100,
              callbacks=[model_save, tensorboard],
              validation_data=(test_images, test_labels))


if __name__ == "__main__":
    main()
