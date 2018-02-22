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
from utils import *


class MNIST():
    """Multilayer perceptron for Problem 1"""

    def __init__(self, layers, learning_rate, lmbda):
        """Initialize multilayer perceptron"""

        self.layers = layers
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.compile()


    def init_weights(self, tensor):
        """Glorot normal weight initialization"""

        if isinstance(tensor, nn.Linear):
            tensor.bias.data.fill_(0)
            nn.init.xavier_normal(tensor.weight.data)

    
    def compile(self):
        """Initialize model parameters"""

        # MLP with 2 hidden layers
        self.model = nn.Sequential(collections.OrderedDict([
                        ("fc1", nn.Linear(self.layers[0], self.layers[1])), 
                        ("relu1", nn.ReLU()),
                        ("fc2", nn.Linear(self.layers[1], self.layers[2])), 
                        ("relu2", nn.ReLU()),
                        ("fc3", nn.Linear(self.layers[2], self.layers[3]))
                     ]))

        # Initialize weights
        self.model.apply(self.init_weights)

        # Set loss function and gradient-descend optimizer
        self.loss_fn = nn.CrossEntropyLoss()
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
            if valid_loader:
                valid_acc.append(self.predict(valid_loader))
            else:
                valid_acc.append(-1)
            if test_loader:
                test_acc.append(self.predict(test_loader))
            else:
                test_acc.append(-1)
          
            # Format printing depending on tracked quantities
            if valid_loader and test_loader: 
                print("Avg loss: {:.4f} -- Train acc: {:.4f} -- Val acc: {:.4f} -- Test acc: {:.4f}".format(train_loss[epoch], train_acc[epoch], valid_acc[epoch], test_acc[epoch]))
        
            if valid_loader and not test_loader:
                print("Avg loss: {:.4f} -- Train acc: {:.4f} -- Val acc: {:.4f}".format(
                    train_loss[epoch], train_acc[epoch], valid_acc[epoch]))

            if not valid_loader and not test_loader:
                print("Avg loss: {:.4f} -- Train acc: {:.4f}  ".format( 
                    train_loss[epoch], train_acc[epoch]))


        # Print elapsed time
        end = datetime.datetime.now()
        elapsed = str(end - start)[:-7]
        print("Training done! Elapsed time: {}\n".format(elapsed))

        return train_loss, train_acc, valid_acc, test_acc


if __name__ == "__main__":

    # Model parameters
    layers = [784, 800, 800, 10]
    learning_rate = 0.02
    lmbda = 2.5
    batch_size = 64
    nb_epochs = 3
    data_filename = "../data/mnist/mnist.pkl"

    # Load data
    train_loader, valid_loader, test_loader = get_data_loaders(data_filename, batch_size)

    # Build MLP and train
    mlp = MNIST(layers, learning_rate, lmbda)
    mlp.train(nb_epochs, train_loader, valid_loader, test_loader)
