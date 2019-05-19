#!/usr/bin/env python3

from time import time
from math import ceil

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

import data.fruit


def main():
    names = data.fruit.load_class_name()
    print(len(names))
    train_gen = data.fruit.generator(100)
    test_gen = data.fruit.generator(100, validation=True)
    # test_gen = data.fruit.generator(1000, True)
    # (train_images, train_labels, _), (test_images, test_labels,
    #                                   _) = data.fruit.load_data()
    # model = keras.Sequential([
    #     keras.layers.Flatten(input_shape=(100, 100, 3)),
    #     keras.layers.Dense(512, activation=tf.nn.relu),
    #     keras.layers.Dense(len(names), activation=tf.nn.softmax)
    # ])
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3),
                            input_shape=(100, 100, 3),
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
        log_dir=".fruit_log/{}".format(time()))
    data_save = keras.callbacks.CSVLogger('.fruit_log/log.csv',
                                          append=True,
                                          separator=',')
    model_save = keras.callbacks.ModelCheckpoint('.fruit_log/{epoch:05}.h5',
                                                 period=10)
    model.summary()
    # model.fit_generator(train_gen, epochs=100)
    # model.fit(train_x, train_y, epochs=10)
    model.fit_generator(train_gen, steps_per_epoch=10,
                        epochs=100,
                        callbacks=[model_save, tensorboard, data_save],
                        validation_data=test_gen)


if __name__ == "__main__":
    main()
