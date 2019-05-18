#!/usr/bin/env python3

from time import time

import tensorflow as tf
from tensorflow import keras
# import numpy as np
# import matplotlib.pyplot as plt

import data.iris

def main():
    (train_images, train_labels, _), (test_images, test_labels,
                                      _) = data.iris.load_data()

if __name__ == "__main__":
    main()
