#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFT6135: Representation Learning
Assignment 2: Convolutional Neural Networks (Problem 2)

Authors:
    Samuel Laferriere <samuel.laferriere.cyr@umontreal.ca>
    Joey Litalien <joey.litalien@umontreal.ca>
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import datetime
import collections
from enum import Enum
from utils import *


class ConvNet(nn.Module):
    """Convolutional neural network"""

    def __init__(self, features):
        super(ConvNet, self).__init__()
        self.arch = [64, 64, 'M',
                     128, 128, 'M',
                     256, 256, 256, 'M',
                     512, 512, 512, 'M',
                     512, 512, 512, 'M']
        self.features = build_layers()
        self.clf = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 2),
        )
        self.init_weights()

    def forward(self, x):
        """Predict"""

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x

    def init_weights(self):
        """Initialize weights and biases (Xavier)"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.WeightNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def build_layers(batch_norm=False, weight_norm=False):
        """Construct deep net (VGG 16)"""

        layers = []
        in_channels = 3
        for v in self.arch:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                elif self.weight_norm:
                    layers += [conv2d, nn.WeightNorm(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)


class CatDog():
    """Deep model for Problem 2"""

    def __init__(self, learning_rate, momentum, batch_norm,
                    weight_norm, weight_decay):
        """Initialize deep net"""

        self.learning_rate = learning_rate
        self.momentum = model_type
        self.batch_norm = batch_norm
        self.weight_norm = weight_norm
        self.weight_decay = weight_decay
        self.compile()

    def compile(self):
        """Initialize model parameters"""

        # Initialize model
        self.model = ConvNet()

        # Set loss function and gradient-descend optimizer
        self.loss_fn = nn.BCELoss()
        self.optimizer = optim.SGD(self.model.parameters(),
                            lr=self.learning_rate,
                            weight_decay=self.weight_decay)

        # CUDA support
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()


    def predict(self, data_loader):
        """Evaluate model on dataset"""

        # Set model phase
        self.model.train(False)

        correct = 0.
        for batch_idx, (x, y) in enumerate(data_loader):
            # Forward pass
            # x, y = Variable(x).view(len(x), 3, 64, 64), Variable(y)
            x, y = Variable(x), Variable(y)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Predict
            y_pred = self.model(x)
            correct += float((y_pred.max(1)[1] == y).sum().data[0]) \
                        / data_loader.batch_size

        # Compute accuracy
        acc = correct / len(data_loader)
        return acc


    def train(self, nb_epochs, train_loader, valid_loader, test_loader):
        """Train model on data"""

        # Set learning phase
        self.model.train(True)

        # Initialize tracked quantities
        train_loss, train_acc, valid_acc, test_acc = [], [], [], []

        # Train
        start = datetime.datetime.now()
        for epoch in range(nb_epochs):
            print("Epoch {:d} | {:d}".format(epoch + 1, nb_epochs))
            losses = AverageMeter()

            # Mini-batch SGD
            for batch_idx, (x, y) in enumerate(train_loader):
                # Print progress bar
                progress_bar(batch_idx, len(train_loader.dataset) / \
                        train_loader.batch_size)

                # Forward pass
                x, y = Variable(x), Variable(y)

                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                # Predict
                y_pred = self.model(x)

                # Compute loss
                loss = self.loss_fn(y_pred, y)
                losses.update(loss.data[0], x.size(0))

                # Zero gradients, perform a backward pass, and update the weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Save losses and accuracies
            train_loss.append(losses.avg)
            train_acc.append(self.predict(train_loader))
            valid_acc.append(self.predict(valid_loader) if valid_loader else -1)
            test_acc.append(self.predict(test_loader) if test_loader else -1)

            # Print statistics
            track = dict(valid = valid_loader is not None,
                         test = test_loader is not None)
            show_learning_stats(track, train_loss[epoch], train_acc[epoch],
                    valid_acc[epoch], test_acc[epoch])

        # Print elapsed time
        end = datetime.datetime.now()
        elapsed = str(end - start)[:-7]
        print("Training done! Elapsed time: {}\n".format(elapsed))

        return train_loss, train_acc, valid_acc, test_acc


if __name__ == "__main__":

    # Model parameters
    nb_epochs = 10
    learning_rate = 0.02
    momentum = 0.9
    batch_norm = False
    weight_norm = False
    weight_decay = 0
    train_dir = "../data/catdog/train"
    valid_dir = "../data/catdog/valid"
    test_dir = "../data/catdog/test"

    # Load data
    train_loader, valid_loader, test_loader = load_catdog(train_dir, valid_dir, test_dir, batch_size)

    # Build deep net and train
    net = CatDog(learning_rate, momentum, batch_norm, weight_norm, weight_decay)
    train_loss, train_acc, valid_acc, test_acc = \
            net.train(nb_epochs, train_loader, valid_loader, test_loader)
