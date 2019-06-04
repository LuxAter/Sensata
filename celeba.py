#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras.layers import *
import data.celeba

from base import ImgMultiClassifier


def main():
    names = data.celeba.load_class_name()
    celeba_class = ImgMultiClassifier(data.celeba)
    if celeba_class.process_vars():
        celeba_class.predict()
        return
    celeba_class.model = keras.Sequential([
        Conv2D(32, (3, 3),
               input_shape=(218, 178, 3),
               padding='same',
               activation='relu'),
        BatchNormalization(),
        Conv2D(32, (3, 3), padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Conv2D(64, (5, 5), padding='same', activation='relu'),
        BatchNormalization(),
        Conv2D(64, (5, 5), padding='same', activation='relu'),
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
        Dense(512, activation='relu'),
        Dense(512, activation='relu'),
        Dense(128, activation='relu'),
        Dense(128, activation='relu'),
        Dense(len(names), activation='softmax')
        ])
    # celeba_class.model = keras.Sequential([
    #     Conv2D(32, (3, 3),
    #            input_shape=(218, 178, 3),
    #            padding='same',
    #            activation='relu'),
    #     BatchNormalization(),
    #     Conv2D(32, (3, 3), padding='same', activation='relu'),
    #     BatchNormalization(),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     # Dropout(0.2),
    #     Conv2D(64, (5, 5), padding='same', activation='relu'),
    #     BatchNormalization(),
    #     Conv2D(64, (5, 5), padding='same', activation='relu'),
    #     BatchNormalization(),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     # Dropout(0.3),
    #     Conv2D(128, (3, 3), padding='same', activation='relu'),
    #     BatchNormalization(),
    #     Conv2D(128, (3, 3), padding='same', activation='relu'),
    #     BatchNormalization(),
    #     MaxPooling2D(pool_size=(2, 2)),
    #     # Dropout(0.4),
    #     Flatten(),
    #     Dense(1024, activation='relu'),
    #     Dense(1024, activation='relu'),
    #     Dense(512, activation='relu'),
    #     Dense(512, activation='relu'),
    #     Dense(len(names), activation='softmax')
    # ])
    celeba_class.model.compile(optimizer='adam',
                               loss='binary_crossentropy',
                               metrics=['accuracy'])
    celeba_class.train()


if __name__ == "__main__":
    main()
