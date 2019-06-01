#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras.layers import *
import data.mnist

from base import ImgClassifier


def main():
    mnist_class = ImgClassifier(data.mnist)
    if mnist_class.process_vars():
        mnist_class.predict()
        return
    mnist_class.model = keras.Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    mnist_class.model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
    mnist_class.train()


if __name__ == "__main__":
    main()
