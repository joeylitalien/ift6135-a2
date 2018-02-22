# -*- coding: utf-8 -*-
"""
IFT6135: Representation Learning
Assignment 2: CNNs, Regularization and Normalization

Authors: 
    Samuel Laferriere <samlaf92@gmail.com>
    Joey Litalien <joey.litalien@mail.mcgill.ca>
"""

from __future__ import print_function

import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pickle


def progress_bar(count, total, status=""):
    """Neat progress bar to track training"""

    bar_size = 20
    filled = int(round(bar_size * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = u"\u25A0" * filled + " " * (bar_size - filled)
    sys.stdout.write("Training [%s] %s%s %s\r" % \
            (bar, percents, "%", status))
    sys.stdout.flush()


def unpickle_mnist(filename):
    """Load data into training/valid/test sets"""

    # Unpickle files (uses latin switch for py2.x to py3.x compatibility)
    if sys.version_info[0] < 3:
        train, valid, test = pickle.load(open(filename, "rb"))
    else:
        train, valid, test = pickle.load(open(filename, "rb"), encoding="latin1")
    X_train, y_train = map(torch.from_numpy, train)
    X_valid, y_valid = map(torch.from_numpy, valid)
    X_test, y_test = map(torch.from_numpy, test)

    # Convert to tensors
    train_data = TensorDataset(X_train, y_train)
    valid_data = TensorDataset(X_valid, y_valid)
    test_data = TensorDataset(X_test, y_test)
    
    return train_data, valid_data, test_data


def get_data_loaders(data_filename, batch_size):
    """Load data from pickled file"""

    train_data, valid_data, test_data = unpickle_mnist(data_filename)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

