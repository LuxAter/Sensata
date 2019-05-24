#!/usr/bin/env python3

from time import time

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

import data.cifar10


def main():
    (train_images, train_labels, _), (test_images, test_labels,
                                      _) = data.cifar10.load_data()
    names = data.cifar10.load_class_name()
    model = keras.Sequential([
        Conv2D(64, (3, 3),
               intput_shape=(32, 32, 3),
               padding='same',
               activation='relu'),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size(2, 2), strides=(2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size(2, 2), strides=(2, 2)),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size(2, 2), strides=(2, 2)),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size(2, 2), strides=(2, 2)),
        Flatten(),
        Dense(4096, activation='relu'),
        Dense(4096, activation='relu'),
        Dense(len(names), activation='relu')
    ])
    # model = keras.Sequential([
    #     keras.layers.Conv2D(32, (3, 3),
    #                         input_shape=(train_images.shape[1:]),
    #                         padding='same'),
    #     keras.layers.Activation('relu'),
    #     keras.layers.Conv2D(32, (3, 3)),
    #     keras.layers.Activation('relu'),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     keras.layers.Dropout(0.25),
    #     keras.layers.Conv2D(64, (3, 3), padding='same'),
    #     keras.layers.Activation('relu'),
    #     keras.layers.Conv2D(64, (3, 3)),
    #     keras.layers.Activation('relu'),
    #     keras.layers.MaxPooling2D(pool_size=(2, 2)),
    #     keras.layers.Dropout(0.25),
    #     keras.layers.Flatten(),
    #     keras.layers.Dense(512),
    #     keras.layers.Activation('relu'),
    #     keras.layers.Dropout(0.5),
    #     keras.layers.Dense(len(names)),
    #     keras.layers.Activation('softmax')
    # ])
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
