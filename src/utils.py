#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFT6135: Representation Learning
Assignment 2: CNNs, Regularization and Normalization

Authors:
    Samuel Laferriere <samuel.laferriere.cyr@umontreal.ca>
    Joey Litalien <joey.litalien@mail.mcgill.ca>
"""

from __future__ import print_function

import os, sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pickle
from matplotlib.ticker import MaxNLocator


class TestImageFolder(Dataset):
    """Test image folder to retrieve image filename from loader"""

    def __init__(self, root, transform=None):
        images = []
        for filename in os.listdir(root):
            if filename.endswith('jpg'):
                images.append('{}'.format(filename))

        self.root = root
        self.imgs = images
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.imgs)


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
    elif m.dropout_masks and m.avg_pre_softmax:
        print("{} (Dropouts / Pre-softmax avg)".format(hdr))
    elif m.dropout_masks and not m.avg_pre_softmax:
        print("{} (Dropouts / Post-softmax avg)".format(hdr))
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


def plot_per_epoch(d, d_label, title):
    """Plot graph; only takes a single list"""

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(1,len(d)+1), d, c="b", s=6, marker="o", label=d_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(d_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    plt.show()

def plots_per_epoch(d, d_labels, tracked_label, title):
    """ Plot graph; takes multiple sets of points """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["b", "r", "g", "m", "c", "y"]
    for i in range(len(d)):
        ax.plot(range(1,len(d[i])+1), d[i], c=colors[i], 
            marker="o", label=d_labels[i])
    plt.legend(loc="lower right")
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(tracked_label)
    ax.set_title(title)
    plt.savefig('batch.png')
    plt.show()


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


def load_mnist(data_filename, batch_size):
    """Load data from pickled file"""

    train_data, valid_data, test_data = unpickle_mnist(data_filename)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, valid_loader, test_loader


def load_catdog(dirs, batch_size):
    """Load data from image folders"""

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_data = ImageFolder(root=dirs["train"],
            transform=transforms.Compose([transforms.Resize((64, 64)),
                                          transforms.RandomRotation(10),
					  transforms.RandomHorizontalFlip(),
                                          transforms.RandomVerticalFlip(),
                                          transforms.ToTensor(),
					  normalize]))

    valid_data = ImageFolder(root=dirs["valid"],
            transform=transforms.Compose([transforms.Resize((64, 64)),
                                          transforms.ToTensor(),
			                  normalize]))

    test_data = ImageFolder(root=dirs["test"],
            transform=transforms.Compose([transforms.Resize((64, 64)),
                                          transforms.ToTensor(),
					  normalize]))

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
