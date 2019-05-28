import os
import glob
import urllib.request
import tarfile
import zipfile
import gzip
import shutil
import pickle
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt


def unpickle(file):
    with open(file, 'rb') as binary:
        return pickle.load(binary, encoding='bytes')
    return None


def class_encoding(classes, num_classes=None):
    if num_classes is None:
        num_classes = np.max(classes) + 1
    return np.eye(num_classes, dtype=float)[classes]


def download_and_extract(dest, url, verbose=True, filename=None):
    filename = url.split('/')[-1] if filename is None else filename
    filepath = os.path.join(dest, filename)
    if not os.path.exists(filepath):
        if not os.path.exists(dest):
            os.makedirs(dest)

        def progress(count, block_size, total_size):
            perc = float(count * block_size) / float(total_size)
            width = 80 - len(">  Downloading {} {:.1f}%".format(
                filename, perc * 100.0))
            if verbose:
                print("\r>> Downloading {} [{}] {:.1f}%".format(
                    filename, ('=' * int(perc * width)) + ">" + (' ' * int(
                        (1.0 - perc) * width)), perc * 100.0),
                      end='')

        filepath, _ = urllib.request.urlretrieve(url=url,
                                                 filename=filepath,
                                                 reporthook=progress)
        if verbose:
            print()
        statinfo = os.stat(filepath)
        if verbose:
            print("   Downloaded", filename, statinfo.st_size, "bytes.")
        if filepath.endswith(".zip"):
            print(">> Extracting {}".format(filepath))
            zipfile.ZipFile(file=filepath, mode="r").extractall(dest)
            print("   Extracted {}".format(filepath))
        elif filepath.endswith((".tar.gz", ".tgz")):
            print(">> Extracting {}".format(filepath))
            tarfile.open(name=filepath, mode="r:gz").extractall(dest)
            print("   Extracted {}".format(filepath))
        elif filepath.endswith(".gz"):
            print(">> Extracting {}".format(filepath))
            with gzip.open(filepath, 'rb') as bin_data:
                with open('.'.join(filepath.split('.')[:-1]), 'wb') as bin_out:
                    shutil.copyfileobj(bin_data, bin_out)
            print("   Extracted {}".format(filepath))


def kaggle_and_extract(dest, url, verbose=True, filename=None):
    filename = url.split('/')[-1] + ".zip" if filename is None else filename
    filepath = os.path.join(dest, filename)
    if not os.path.exists(filepath):
        if not os.path.exists(dest):
            os.makedirs(dest)
        os.system("kaggle datasets download -d {} -p {}".format(url, dest))
        if verbose:
            print()
        statinfo = os.stat(filepath)
        if verbose:
            print("   Downloaded", filename, statinfo.st_size, "bytes.")
        if filepath.endswith(".zip"):
            print(">> Extracting {}".format(filepath))
            zipfile.ZipFile(file=filepath, mode="r").extractall(dest)
            print("   Extracted {}".format(filepath))
        elif filepath.endswith((".tar.gz", ".tgz")):
            print(">> Extracting {}".format(filepath))
            tarfile.open(name=filepath, mode="r:gz").extractall(dest)
            print("   Extracted {}".format(filepath))
        elif filepath.endswith(".gz"):
            print(">> Extracting {}".format(filepath))
            with gzip.open(filepath, 'rb') as bin_data:
                with open('.'.join(filepath.split('.')[:-1]), 'wb') as bin_out:
                    shutil.copyfileobj(bin_data, bin_out)
            print("   Extracted {}".format(filepath))


def kaggle_and_extract_all(dest, url, verbose=True, filename=None):
    filename = url.split('/')[-1] + ".zip" if filename is None else filename
    filepath = os.path.join(dest, filename)
    if not os.path.exists(filepath):
        if not os.path.exists(dest):
            os.makedirs(dest)
        os.system("kaggle datasets download -d {} -p {}".format(url, dest))
        if verbose:
            print()
        statinfo = os.stat(filepath)
        if verbose:
            print("   Downloaded", filename, statinfo.st_size, "bytes.")
        if filepath.endswith(".zip"):
            print(">> Extracting {}".format(filepath))
            zipfile.ZipFile(file=filepath, mode="r").extractall(dest)
            print("   Extracted {}".format(filepath))
        elif filepath.endswith((".tar.gz", ".tgz")):
            print(">> Extracting {}".format(filepath))
            tarfile.open(name=filepath, mode="r:gz").extractall(dest)
            print("   Extracted {}".format(filepath))
        elif filepath.endswith(".gz"):
            print(">> Extracting {}".format(filepath))
            with gzip.open(filepath, 'rb') as bin_data:
                with open('.'.join(filepath.split('.')[:-1]), 'wb') as bin_out:
                    shutil.copyfileobj(bin_data, bin_out)
            print("   Extracted {}".format(filepath))
        files = glob.glob(dest + '/*.zip') + glob.glob(
            dest + '/*.tar.gz') + glob.glob(dest + "/*.gz")
        for file in files:
            if file == filepath:
                continue
            elif file.endswith(".zip"):
                print(">> Extracting {}".format(file))
                zipfile.ZipFile(file=file, mode="r").extractall(dest)
                print("   Extracted {}".format(file))
            elif file.endswith((".tar.gz", ".tgz")):
                print(">> Extracting {}".format(file))
                tarfile.open(name=file, mode="r:gz").extractall(dest)
                print("   Extracted {}".format(file))
            elif file.endswith(".gz"):
                print(">> Extracting {}".format(file))
                with gzip.open(file, 'rb') as bin_data:
                    with open('.'.join(file.split('.')[:-1]), 'wb') as bin_out:
                        shutil.copyfileobj(bin_data, bin_out)
                print("   Extracted {}".format(file))


class Generator(keras.utils.Sequence):

    def __init__(self,
                 files,
                 labels,
                 batch_size=32,
                 dim=(32, 32),
                 n_classes=10,
                 shuffle=True):
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.files = files
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.files) / self.batch_size))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size:(idx + 1) *
                               self.batch_size]
        # files_tmp = [self.files[k] for k in indexes]
        x, y = self.__data(indexes)
        return x, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data(self, ids):
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size), dtype=int)
        for i, idx in enumerate(ids):
            x[i,] = plt.imread(self.files[idx]).astype(np.float64) / 255.0
            y[i] = self.labels[idx]
        return x, y
