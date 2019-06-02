#!/usr/bin/env python3

from os import makedirs
from os.path import basename
from sys import argv
from time import time
from pathlib import Path

import tarfile

from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np


class ImgClassifier(object):

    def __init__(self, data, max_epoch=100, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.model = None
        self.start_time = int(time())
        self.should_zip = True
        self.file_path = ".{}_log/{}".format(argv[0].strip("./").strip(".py"),
                                             self.start_time)
        makedirs(self.file_path)
        self.callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.file_path),
            keras.callbacks.CSVLogger('{}/log.csv'.format(self.file_path),
                                      append=True,
                                      separator=','),
            keras.callbacks.ModelCheckpoint('{}/{{epoch}}.h5'.format(
                self.file_path),
                                            period=10)
        ]

    def process_vars(self):
        if len(argv) > 1 and argv[1].endswith('.h5'):
            return True
        elif len(argv) > 1 and argv[1].isdigit():
            self.max_epoch = int(argv[1])
        return False

    def predict(self):
        names = self.data.load_class_name()
        if self.model is None:
            self.model = keras.models.load_model(argv[1])
        count = int(argv[2]) if len(argv) > 2 else 10
        imgs, labels, _ = self.data.load_test_data()
        if isinstance(imgs[0], Path):
            files = [str(x) for x in imgs]
            for i in range(count):
                index = np.random.randint(0, len(files))
                img = plt.imread(files[index]).astype(np.float64) / 255.0
                key = labels[index]
                res = self.model.predict(np.expand_dims(img,
                                                        axis=0)).tolist()[0]
                plt.imshow(img)
                plt.title("{}[{}]".format(names[res.index(max(res))],
                                          names[key]))
                plt.show()
        else:
            for i in range(count):
                index = np.random.randint(0, len(imgs))
                img = imgs[index]
                key = labels[index]
                res = self.model.predict(np.expand_dims(img,
                                                        axis=0)).tolist()[0]
                plt.imshow(img)
                plt.title("{}[{}]".format(names[res.index(max(res))],
                                          names[key]))
                plt.show()

    def train(self):
        with open("{}/summary.txt".format(self.file_path), 'w') as file:
            self.model.summary(print_fn=lambda x: file.write(x + '\n'))
        if hasattr(self.data, 'generator'):
            self.train_generator()
        else:
            self.train_data()
        self.model.save("{}/final.h5".format(self.file_path))
        if self.should_zip:
            self.zip()

    def train_generator(self):
        train_gen = self.data.generator(self.batch_size)
        test_gen = self.data.generator(self.batch_size, validation=True)
        if test_gen:
            self.model.fit_generator(train_gen,
                                     epochs=self.max_epoch,
                                     callbacks=self.callbacks,
                                     validation_data=test_gen)
        else:
            self.model.fit_generator(train_gen,
                                     epochs=self.max_epoch,
                                     callbacks=self.callbacks)

    def train_data(self):
        train_set, test_set = self.data.load_data()
        if test_set:
            self.model.fit(train_set[0],
                           train_set[1],
                           epochs=self.max_epoch,
                           callbacks=self.callbacks,
                           validation_data=(test_set[0], test_set[1]))
        else:
            self.model.fit(train_set[0],
                           train_set[1],
                           epochs=self.max_epoch,
                           callbacks=self.callbacks)

    def zip(self):
        with tarfile.open("{}.tar.gz".format(self.file_path.strip('/')),
                          "w:gz") as tar:
            tar.add(self.file_path, arcname=basename(self.file_path))


class ImgMultiClassifier(object):

    def __init__(self, data, max_epoch=100, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.model = None
        self.start_time = int(time())
        self.should_zip = True
        self.file_path = ".{}_log/{}".format(argv[0].strip("./").strip(".py"),
                                             self.start_time)
        makedirs(self.file_path)
        self.callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.file_path),
            keras.callbacks.CSVLogger('{}/log.csv'.format(self.file_path),
                                      append=True,
                                      separator=','),
            keras.callbacks.ModelCheckpoint('{}/{{epoch}}.h5'.format(
                self.file_path),
                                            period=10)
        ]

    def process_vars(self):
        if len(argv) > 1 and argv[1].endswith('.h5'):
            return True
        elif len(argv) > 1 and argv[1].isdigit():
            self.max_epoch = int(argv[1])
        return False

    def predict(self):
        names = self.data.load_class_name()
        if self.model is None:
            self.model = keras.models.load_model(argv[1])
        count = int(argv[2]) if len(argv) > 2 else 10
        imgs, labels, _ = self.data.load_test_data()
        if isinstance(imgs[0], Path):
            files = [str(x) for x in imgs]
            for i in range(count):
                index = np.random.randint(0, len(files))
                img = plt.imread(files[index]).astype(np.float64) / 255.0
                key = labels[index]
                res = self.model.predict(np.expand_dims(img,
                                                        axis=0)).tolist()[0]
                plt.imshow(img)
                res_vals = ",".join(
                    [names[j] for j, v in enumerate(res) if v >= 0.85])
                key_vals = ",".join(
                    [names[j] for j, v in enumerate(key) if v >= 0.85])
                plt.title("{}[{}]".format(res_vals, key_vals))
                plt.show()
        else:
            for i in range(count):
                index = np.random.randint(0, len(imgs))
                img = imgs[index]
                key = labels[index]
                res = self.model.predict(np.expand_dims(img,
                                                        axis=0)).tolist()[0]
                plt.imshow(img)
                res_vals = ",".join(
                    [names[j] for j, v in enumerate(res) if v >= 0.85])
                key_vals = ",".join(
                    [names[j] for j, v in enumerate(key) if v >= 0.85])
                plt.title("{}[{}]".format(res_vals, key_vals))
                plt.show()

    def train(self):
        with open("{}/summary.txt".format(self.file_path), 'w') as file:
            self.model.summary(print_fn=lambda x: file.write(x + '\n'))
        if hasattr(self.data, 'generator'):
            self.train_generator()
        else:
            self.train_data()
        self.model.save("{}/final.h5".format(self.file_path))
        if self.should_zip:
            self.zip()

    def train_generator(self):
        train_gen = self.data.generator(self.batch_size)
        test_gen = self.data.generator(self.batch_size, validation=True)
        if test_gen:
            self.model.fit_generator(train_gen,
                                     epochs=self.max_epoch,
                                     callbacks=self.callbacks,
                                     validation_data=test_gen)
        else:
            self.model.fit_generator(train_gen,
                                     epochs=self.max_epoch,
                                     callbacks=self.callbacks)

    def train_data(self):
        train_set, test_set = self.data.load_data()
        if test_set:
            self.model.fit(train_set[0],
                           train_set[1],
                           epochs=self.max_epoch,
                           callbacks=self.callbacks,
                           validation_data=(test_set[0], test_set[1]))
        else:
            self.model.fit(train_set[0],
                           train_set[1],
                           epochs=self.max_epoch,
                           callbacks=self.callbacks)

    def zip(self):
        with tarfile.open("{}.tar.gz".format(self.file_path.strip('/')),
                          "w:gz") as tar:
            tar.add(self.file_path, arcname=basename(self.file_path))
