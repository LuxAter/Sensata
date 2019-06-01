#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras.layers import *
import data.fruit

from base import ImgClassifier


def main():
    names = data.fruit.load_class_name()
    fruit_class = ImgClassifier(data.fruit)
    if fruit_class.process_vars():
        fruit_class.predict()
        return
    fruit_class.model = keras.Sequential([
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
        Dense(len(names) * 2, activation='relu'),
        Dense(len(names), activation='softmax')
    ])
    fruit_class.model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
    fruit_class.train()


if __name__ == "__main__":
    main()
