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
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=".mnist_log/{}".format(time()))
    data_save = keras.callbacks.CSVLogger('.mnist_log/log.csv', append=True, separator=',')
    model_save = keras.callbacks.ModelCheckpoint('.mnist_log/{epoch:05}.h5',
                                                 period=10)
    model.fit(train_images,
              train_labels,
              epochs=100,
              callbacks=[model_save, tensorboard, data_save],
              validation_data=(test_images, test_labels))


if __name__ == "__main__":
    main()
