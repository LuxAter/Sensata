import os
import urllib.request
import tarfile
import zipfile
import gzip
import shutil
import pickle
import numpy as np


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
