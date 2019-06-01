#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras.layers import *
import data.cifar10

from base import ImgClassifier


def main():
    names = data.cifar10.load_class_name()
    cifar10_class = ImgClassifier(data.cifar10)
    if cifar10_class.process_vars():
        cifar10_class.predict()
        return
    cifar10_class.model = keras.Sequential([
        Conv2D(32, (3, 3),
               input_shape=(32, 32, 3),
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
        Dense(256, activation='relu'),
        Dense(len(names), activation='softmax')
    ])
    cifar10_class.model.compile(optimizer='adam',
                                loss='sparse_categorical_crossentropy',
                                metrics=['accuracy'])
    cifar10_class.train()


if __name__ == "__main__":
    main()
