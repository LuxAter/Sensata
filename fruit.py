#!/usr/bin/env python3

from time import time

import tensorflow as tf
from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

import data.fruit

def main():
    (train_images, train_labels, _), (test_images, test_labels,
                                      _) = data.fruit.load_data()

if __name__ == "__main__":
    main()
