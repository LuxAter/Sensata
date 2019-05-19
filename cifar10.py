#!/usr/bin/env python3

from time import time

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import data.cifar10


def main():
    (train_images, train_labels, _), (test_images, test_labels,
                                      _) = data.cifar10.load_data()
    names = data.cifar10.load_class_name()
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3),
                            input_shape=(train_images.shape[1:]),
                            padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(32, (3, 3)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.Activation('relu'),
        keras.layers.Conv2D(64, (3, 3)),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(512),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(len(names)),
        keras.layers.Activation('softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=".cifar10_log/{}".format(time()))
    data_save = keras.callbacks.CSVLogger('.cifar10_log/log.csv',
                                          append=True,
                                          separator=',')
    model_save = keras.callbacks.ModelCheckpoint('.cifar10_log/{epoch:05}.h5',
                                                 period=10)
    print("HELLO")
    print(train_images, train_labels)
    model.fit(train_images,
              train_labels,
              epochs=100,
              callbacks=[model_save, tensorboard, data_save],
              validation_data=(test_images, test_labels))


if __name__ == "__main__":
    main()
