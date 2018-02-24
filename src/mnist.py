#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFT6135: Representation Learning
Assignment 2: CNNs, Regularization and Normalization (Problem 1)

Authors: 
    Samuel Laferriere <samlaf92@gmail.com>
    Joey Litalien <joey.litalien@mail.mcgill.ca>
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import datetime
import collections
from enum import Enum
from utils import *


class MLP(nn.Module):
    """Multilayer perceptron"""

    def __init__(self, weight_decay=0, dropout=0):
        super(MLP, self).__init__()
        self.model = nn.Sequential(collections.OrderedDict([
            # Layer 1
            ("fc_1", nn.Linear(784, 800)), 
            ("relu_1", nn.ReLU()),

            # Layer 2
            ("fc_2", nn.Linear(800, 800)), 
            ("relu_2", nn.ReLU()),

            # Output layer
            ("fc_3", nn.Linear(800, 10))
        ]))

    def forward(self, x):
        return self.model(x)


class CNN(nn.Module):
    """Convolutional neural network"""

    def __init__(self, batch_norm=False):
        super(CNN, self).__init__()
        if (batch_norm):
            self.model = nn.Sequential(collections.OrderedDict([
                # Layer 1
                ("conv2d_1", nn.Conv2d(in_channels=1, out_channels=16, 
                    kernel_size=(3, 3), padding=1)),
                ("batchnorm2d_1", nn.BatchNorm2d(16)),
                ("dropout_1", nn.Dropout(p=0.5)),
                ("relu_1", nn.ReLU()),
                ("maxpool2d_1", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                
                # Layer 2
                ("conv2d_2", nn.Conv2d(in_channels=16, out_channels=32, 
                    kernel_size=(3, 3), padding=1)),
                ("batchnorm2d_2", nn.BatchNorm2d(32)),
                ("dropout_2", nn.Dropout(p=0.5)),
                ("relu_2", nn.ReLU()),
                ("maxpool2d_2", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                
                # Layer 3
                ("conv2d_3", nn.Conv2d(in_channels=32, out_channels=64, 
                    kernel_size=(3, 3), padding=1)),
                ("batchnorm2d_3", nn.BatchNorm2d(64)),
                ("dropout_3", nn.Dropout(p=0.5)),
                ("relu_3", nn.ReLU()),
                ("maxpool2d_3", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                
                # Layer 4
                ("conv2d_4", nn.Conv2d(in_channels=64, out_channels=128, 
                    kernel_size=(3, 3), padding=1)),
                ("batchnorm2d_4", nn.BatchNorm2d(128)),
                ("dropout_4", nn.Dropout(p=0.5)),
                ("relu_4", nn.ReLU()),
                ("maxpool2d_4", nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            ]))
        
        else:
            self.model = nn.Sequential(collections.OrderedDict([
                # Layer 1
                ("conv2d_1", nn.Conv2d(in_channels=1, out_channels=16, 
                    kernel_size=(3, 3), padding=1)),
                ("dropout_1", nn.Dropout(p=0.5)),
                ("relu_1", nn.ReLU()),
                ("maxpool2d_1", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                
                # Layer 2
                ("conv2d_2", nn.Conv2d(in_channels=16, out_channels=32, 
                    kernel_size=(3, 3), padding=1)),
                ("dropout_2", nn.Dropout(p=0.5)),
                ("relu_2", nn.ReLU()),
                ("maxpool2d_2", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                
                # Layer 3
                ("conv2d_3", nn.Conv2d(in_channels=32, out_channels=64, 
                    kernel_size=(3, 3), padding=1)),
                ("dropout_3", nn.Dropout(p=0.5)),
                ("relu_3", nn.ReLU()),
                ("maxpool2d_3", nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                
                # Layer 4
                ("conv2d_4", nn.Conv2d(in_channels=64, out_channels=128, 
                    kernel_size=(3, 3), padding=1)),
                ("dropout_4", nn.Dropout(p=0.5)),
                ("relu_4", nn.ReLU()),
                ("maxpool2d_4", nn.MaxPool2d(kernel_size=(2, 2), stride=2))
            ]))

        # Output layer
        self.clf = nn.Linear(128, 10) 

    def forward(self, x):
        return self.clf(self.model(x).squeeze())


class MNIST():
    """Deep model for Problem 1"""

    def __init__(self, learning_rate, lmbda, model_type, weight_decay, dropout, batch_norm):
        """Initialize multilayer perceptron"""

        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.model_type = model_type
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.compile()


    def init_weights(self, tensor):
        """Glorot normal weight initialization"""

        if isinstance(tensor, nn.Linear):
            tensor.bias.data.fill_(0)
            nn.init.xavier_normal(tensor.weight.data)

    
    def compile(self):
        """Initialize model parameters"""

        # Choose type of model
        if self.model_type == Net.CNN:
            self.model = CNN(self.batch_norm)
        else:
            self.model = MLP(self.weight_decay, self.dropout)

        # Initialize weights
        self.model.apply(self.init_weights)

        # Set loss function and gradient-descend optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        if self.weight_decay != 0:
            self.optimizer = optim.SGD(self.model.parameters(), 
                            lr=self.learning_rate,
                            weight_decay=self.weight_decay)

        else:
            self.optimizer = optim.SGD(self.model.parameters(), 
                            lr=self.learning_rate)

        # CUDA support
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()


    def predict(self, data_loader):
        """Evaluate model on dataset"""

        correct = 0.
        for batch_idx, (x, y) in enumerate(data_loader):
            # Forward pass
            if (self.model_type == Net.CNN):
                x, y = Variable(x).view(len(x), 1, 28, 28), Variable(y)
            else:
                x, y = Variable(x).view(len(x), -1), Variable(y)

            if torch.cuda.is_available(): 
                x = x.cuda()
                y = y.cuda()
            
            # Predict
            y_pred = self.model(x)
            correct += float((y_pred.max(1)[1] == y).sum().data[0]) / data_loader.batch_size 

        # Compute accuracy
        acc = correct / len(data_loader)
        return acc 


    def train(self, nb_epochs, train_loader, valid_loader, test_loader):
        """Train model on data"""

        # Format header
        format_header(self)

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
                progress_bar(batch_idx, len(train_loader.dataset) / train_loader.batch_size)

                # Forward pass
                if (self.model_type == Net.CNN):
                    x, y = Variable(x).view(len(x), 1, 28, 28), Variable(y)
                else:
                    x, y = Variable(x).view(len(x), -1), Variable(y)

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
            track = { "valid": valid_loader is not None, 
                      "test": test_loader is not None }
            show_learning_stats(track, train_loss[epoch], train_acc[epoch], 
                valid_acc[epoch], test_acc[epoch])
         
        # Print elapsed time
        end = datetime.datetime.now()
        elapsed = str(end - start)[:-7]
        print("Training done! Elapsed time: {}\n".format(elapsed))

        return train_loss, train_acc, valid_acc, test_acc


if __name__ == "__main__":

    # Enum to choose which model to train
    class Net(Enum):
        MLP = 0
        CNN = 1

    # Model parameters
    learning_rate = 0.02
    lmbda = 2.5
    batch_size = 64
    nb_epochs = 3
    model_type = Net.CNN
    weight_decay = 2.5
    dropout = 10
    batch_norm = False
    data_filename = "../data/mnist/mnist.pkl"

    # Load data
    train_loader, valid_loader, test_loader = get_data_loaders(data_filename, batch_size)

    # Build MLP and train
    mlp = MNIST(learning_rate, lmbda, model_type, weight_decay, dropout, batch_norm)
    mlp.train(nb_epochs, train_loader, valid_loader, test_loader)
