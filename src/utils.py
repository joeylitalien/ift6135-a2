#!/usr/bin/env python3
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


def format_header(m):
    """Specify model used"""

    models = ["Multilayer Perceptron", "Convolutional Neural Network"]
    regs_norms = ["L2 regularization", "Batch normalization"]
    hdr = models[m.model_type.value]
    if m.weight_decay != 0:
        print("{} ({})".format(hdr, "L2 regularization"))
    elif m.n_dropout != 0 and m.avg_pre_softmax:
        print("{} (Dropout N={:d} / Pre-softmax avg)".format(hdr, m.n_dropout))
    elif m.n_dropout != 0 and not m.avg_pre_softmax:
        print("{} (Dropout N={:d} / Post-softmax avg)".format(hdr, m.n_dropout))
    elif m.batch_norm:
        print("{} ({})".format(hdr, "Batch normalization"))
    else:
        print("{} ({})".format(hdr, "Vanilla"))


def show_learning_stats(track, train_loss, train_acc, valid_acc, test_acc):
    """Format printing depending on tracked quantities"""

    if track["valid"] and track["test"]: 
        print("Train loss: {:.4f} -- Train acc: {:.4f} -- Val acc: {:.4f} -- Test acc: {:.4f}".format(
            train_loss, train_acc, valid_acc, test_acc))

    if track["valid"] and not track["test"]:
        print("Train loss: {:.4f} -- Train acc: {:.4f} -- Val acc: {:.4f}".format(
            train_loss, train_acc, valid_acc))

    if not track["valid"] and track["test"]:
        print("Train loss: {:.4f} -- Train acc: {:.4f} -- Test acc: {:.4f}".format(
            train_loss, train_acc, test_acc))

    if not track["valid"] and not track["test"]:
        print("Train loss: {:.4f} -- Train acc: {:.4f}  ".format(
            train_loss, train_acc))


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
    """
    Computes and stores the average and current value
    Useful for keeping track of training loss
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0.
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
