#!/usr/bin/env python3

import sys

from time import time
from math import ceil

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np
import matplotlib.pyplot as plt

import data.fruit

def predicter():
    names = data.fruit.load_class_name()
    model = keras.models.load_model(sys.argv[1])
    files, labels, _ = data.fruit.load_test_data()
    files = [str(x) for x in files]
    while True:
        file_path = np.random.choice(files)
        index = files.index(file_path)
        img = plt.imread(files[index]).astype(np.float64) / 255.0
        key = labels[index]
        res = list(model.predict(np.expand_dims(img, axis=0)))
        loss = model.evaluate(np.expand_dims(img, axis=0), np.expand_dims(labels[index], axis=0))
        plt.imshow(img)
        plt.title("{}[{}]".format(names[res.index(max(res))], names[key]))
        plt.show()


def main():
    names = data.fruit.load_class_name()
    train_gen = data.fruit.generator(256)
    test_gen = data.fruit.generator(256, validation=True)

    model = keras.Sequential([
        Conv2D(32, (3, 3),
               input_shape=(100, 100, 3),
               padding='same',
               activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.3),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(128, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),
        Flatten(),
        Dense(len(names)*2, activation='relu'),
        Dense(len(names), activation='softmax')
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
        '{}/{{epoch:05}}.h5'.format(file_path), period=50)
    model.summary()
    model.fit_generator(train_gen,
                        epochs=100,
                        callbacks=[tensorboard, model_save, data_save],
                        validation_data=test_gen)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predicter()
    else:
        main()
