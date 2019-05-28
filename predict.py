#!/usr/bin/env python3

import sys

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def main():
    model = keras.models.load_model(sys.argv[1])
    while True:
        file_path = input("Test File >>")
        if file_path in ('quit', 'exit'):
            break
        img = plt.imread(file_path).astype(np.float64) / 255.0
        plt.imshow(img)
        model.predict(img)
        plt.show()

if __name__ == "__main__":
    main()
