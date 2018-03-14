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
from torch.nn.utils import weight_norm

import numpy as np
import datetime
import collections
from enum import Enum
from utils import *
import math
import sys
import matplotlib.pyplot as plt


class ConvNet(nn.Module):
    """Convolutional neural network"""

    def __init__(self, params):
        super(ConvNet, self).__init__()
	
	# Model architecture
        self.arch = [16, 16, 'M',
                     32, 32, 'M',
                     64, 64, 64, 'M',
                     128, 128, 128, 'M']
	self.dropout = params["dropout"]
        self.batch_norm = params["batch_norm"]
	self.weight_norm = params["weight_norm"]
        self.features = self.build_layers()

	# Dropout
	if self.dropout:
	    self.clf = nn.Sequential(
		nn.Linear(2048, 1024),
		nn.ReLU(inplace=True),
		nn.Dropout(p=0.5),
		nn.Linear(1024, 512),
		nn.ReLU(inplace=True),
		nn.Dropout(p=0.5),
		nn.Linear(512, 2)
	    )
	else:
	    self.clf = nn.Sequential(
		nn.Linear(2048, 512),
		nn.ReLU(inplace=True),
		nn.Linear(512, 2)
	    )

	# He initialization
        self.init_weights()

	# Weight normalization
	if self.weight_norm:
	    for m in self.modules():
                if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
		    m.apply(nn.utils.weight_norm)	

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x

    def init_weights(self):
        """Initialize weights and biases (He, 2015)"""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal(m.weight.data)
                m.bias.data.zero_()

    def build_layers(self):
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
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)


class CatDog():
    """Deep model for Problem 2"""

    def __init__(self, params):
        """Initialize deep net"""

        self.batch_size = params["batch_size"]
	self.train_len = params["train_len"]
        self.learning_rate = params["learning_rate"]
        self.momentum = params["momentum"]
	self.dropout = params["dropout"]
        self.batch_norm = params["batch_norm"]
        self.weight_norm = params["weight_norm"]
        self.weight_decay = params["weight_decay"]
        self.optim = params["optim"]
        self.compile()

    def compile(self):
        """Initialize model parameters"""

	if self.batch_norm and self.weight_norm:
	    print("Error! Can't have both batch and weight norm")
	    sys.exit()

        # Initialize model
	params = dict(dropout=self.dropout,
		      batch_norm=self.batch_norm, 
		      weight_norm=self.weight_norm)
        self.model = ConvNet(params)

        # Set loss function and gradient-descend optimizer
        self.loss_fn = nn.CrossEntropyLoss()
	rescaled_lambda = self.weight_decay * self.batch_size / self.train_len
        if self.optim is "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),
                            lr=self.learning_rate,
			    weight_decay=rescaled_lambda)
        elif self.optim is "RMSProp":
            self.optimizer = optim.SGD(self.model.parameters(),
                            lr=self.learning_rate,
			    weight_decay=rescaled_lambda)
        else:
            self.optimizer = optim.Adam(self.model.parameters(),
                            lr=self.learning_rate,
			    weight_decay=rescaled_lambda)

        # CUDA support
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()


    def top_misclassified_per_batch(self, x, y, y_pred):
	"""Get top misclassified examples in a batch"""

	misclassified_idx = torch.nonzero(y_pred.max(1)[1] != y).view(-1).data
	probs = nn.Softmax(dim=1)(y_pred).max(1)[0].data
	misclassified_probs = probs.index_select(0, misclassified_idx)
	top_prob = misclassified_probs.max(0)[0]
	top_idx = misclassified_probs.max(0)[1]
	return top_idx, top_prob

    def most_uncertain_per_batch(self, y_pred):
	"""Get most uncertain example (pred ~= 50%) in a batch"""
	preds = (y_pred[:,0] - .5).abs()
	most_uncertain_idx = preds.min(0)[1].data[0]
	most_uncertain_prob = preds[most_uncertain_idx].data[0] + .5
	return most_uncertain_idx, most_uncertain_prob
	


    def predict(self, data_loader, test=False):
        """Evaluate model on dataset"""

        # Set model phase
        self.model.train(False)

        correct = 0.
	top_prob, top_img = 0., torch.zeros(64,64,3)
	mu_prob, mu_img = 1, torch.zeros(64,64,3)
        for batch_idx, (x, y) in enumerate(data_loader):
            # Forward pass
            x, y = Variable(x), Variable(y)

            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Predicts
            y_pred = self.model(x)

	    # Get top misclassified example in this batch
	    if test:
		idx, prob = self.top_misclassified_per_batch(x, y, y_pred)
		if prob[0] > top_prob:
		    top_idx = idx
		    top_prob = prob[0]
		    top_img = x[idx].view(3,64,64).data.cpu().permute(2,1,0)

		idx, prob = self.most_uncertain_per_batch(y_pred)
		if prob < mu_prob:
		    mu_idx = idx
		    mu_prob = prob
		    mu_img = x[idx].view(3,64,64).data.cpu().permute(2,1,0)

	    # Accuracy meter
            correct += float((y_pred.max(1)[1] == y).sum().data[0]) / data_loader.batch_size
	
	if test:
		plt.imsave('misclassified_{:.4f}.png'.format(top_prob), top_img)
		plt.imsave('uncertain_{:.4f}.png'.format(mu_prob), mu_img)

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
            train_acc.append(self.predict(train_loader, test=False))
            valid_acc.append(self.predict(valid_loader, test=False) if valid_loader else -1)
            test_acc.append(self.predict(test_loader, test=False) if test_loader else -1)

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
    params = dict(
    	nb_epochs = 100,
    	batch_size = 50,
	train_len = 15000,
    	learning_rate = 0.001,
    	momentum = 0.9,
	dropout = True,
    	batch_norm = False,
    	weight_norm = False,
    	weight_decay = 0,
        optim = "Adam"
    )

    # Image folders
    dirs = dict(
    	train = "../data/catdog/train",
    	valid = "../data/catdog/valid",
    	test = "../data/catdog/test"
    )

    # Load data
    train_loader, valid_loader, test_loader = load_catdog(dirs, params["batch_size"])

    # Build deep net and train
    #net = CatDog(params)
    #train_loss, train_acc, valid_acc, test_acc = \
    #        net.train(params["nb_epochs"], train_loader, valid_loader, test_loader)
    #data = dict(train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)
    #with open('dropout.pkl', 'wb') as fp:
    #	pickle.dump(data, fp)

    params["batch_norm"] = False
    params["weight_norm"] = False
    params["optim"] = "SGD"
    params["learning_rate"] = 0.001
    net = CatDog(params)
    print(params)
    train_loss, train_acc, valid_acc, test_acc = \
            net.train(params["nb_epochs"], train_loader, valid_loader, test_loader)
    data = dict(train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)
    with open('stats/vanilla.pkl', 'wb') as fp:
	pickle.dump(data, fp)

    params["batch_norm"] = True
    params["weight_norm"] = False
    params["optim"] = "SGD"
    params["learning_rate"] = 0.001
    print(params)
    net = CatDog(params)
    train_loss, train_acc, valid_acc, test_acc = \
            net.train(params["nb_epochs"], train_loader, valid_loader, test_loader)
    data = dict(train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)
    with open('stats/sgd_bn_lr_0.001.pkl', 'wb') as fp:
	pickle.dump(data, fp)

    params["batch_norm"] = True
    params["weight_norm"] = False
    params["optim"] = "Adam"
    params["learning_rate"] = 0.001
    print(params)
    net = CatDog(params)
    train_loss, train_acc, valid_acc, test_acc = \
            net.train(params["nb_epochs"], train_loader, valid_loader, test_loader)
    data = dict(train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)
    with open('stats/adam_bn_lr_0.001.pkl', 'wb') as fp:
	pickle.dump(data, fp)

    params["batch_norm"] = True
    params["weight_norm"] = False
    params["optim"] = "RMSProp"
    params["learning_rate"] = 0.001
    print(params)
    net = CatDog(params)
    train_loss, train_acc, valid_acc, test_acc = \
            net.train(params["nb_epochs"], train_loader, valid_loader, test_loader)
    data = dict(train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)
    with open('stats/rmsprop_bn_lr_0.001.pkl', 'wb') as fp:
	pickle.dump(data, fp)

    params["batch_norm"] = False
    params["weight_norm"] = True
    params["learning_rate"] = 0.001
    params["optim"] = "Adam"
    print(params)
    net = CatDog(params)
    train_loss, train_acc, valid_acc, test_acc = \
            net.train(params["nb_epochs"], train_loader, valid_loader, test_loader)
    data = dict(train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)
    with open('stats/adam_wn_lr_0.001.pkl', 'wb') as fp:
	pickle.dump(data, fp)

    params["batch_norm"] = True
    params["weight_norm"] = False
    params["learning_rate"] = 0.0001
    params["optim"] = "Adam"
    print(params)
    net = CatDog(params)
    train_loss, train_acc, valid_acc, test_acc = \
            net.train(params["nb_epochs"], train_loader, valid_loader, test_loader)
    data = dict(train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)
    with open('stats/adam_bn_lr_0.0001.pkl', 'wb') as fp:
	pickle.dump(data, fp)

    params["batch_norm"] = True
    params["weight_norm"] = False
    params["learning_rate"] = 0.01
    params["optim"] = "Adam"
    print(params)
    net = CatDog(params)
    train_loss, train_acc, valid_acc, test_acc = \
            net.train(params["nb_epochs"], train_loader, valid_loader, test_loader)
    data = dict(train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)
    with open('stats/adam_bn_lr_0.01.pkl', 'wb') as fp:
	pickle.dump(data, fp)

    params["batch_norm"] = True
    params["weight_norm"] = False
    params["learning_rate"] = 0.001
    params["weight_decay"] = 0.5
    params["optim"] = "Adam"
    print(params)
    net = CatDog(params)
    train_loss, train_acc, valid_acc, test_acc = \
            net.train(params["nb_epochs"], train_loader, valid_loader, test_loader)
    data = dict(train_loss=train_loss, train_acc=train_acc, valid_acc=valid_acc, test_acc=test_acc)
    with open('stats/adam_weight_decay_0.001_lr_0.001.pkl', 'wb') as fp:
	pickle.dump(data, fp)


    """
    with open("weight_norm_dp.pkl", "rb") as fp:
	data = pickle.load(fp)

    plots_per_epoch([data["train_acc"], data["valid_acc"]], ["Train", "Valid"], "Accuracy", "Dogs vs. Cats Accuracy for Batch Norm")
    """

    
