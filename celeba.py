#!/usr/bin/env python3

from time import time

import tensorflow as tf
from tensorflow import keras
# import numpy as np
import matplotlib.pyplot as plt

import data.celeba

def main():
    train_gen = data.celeba.generator(64)
    x, y = train_gen.__getitem__(0)
    for i in range(len(x)):
        plt.imshow(x[i])
        plt.show()
    # test_gen = data.celeba.generator(256, validation=True)

if __name__ == "__main__":
    main()
