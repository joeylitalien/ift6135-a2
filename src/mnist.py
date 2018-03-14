#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IFT6135: Representation Learning
Assignment 2: CNNs, Regularization and Normalization (Problem 1)

Authors:
	Samuel Laferriere <samuel.laferriere.cyr@umontreal.ca>
	Joey Litalien <joey.litalien@mail.mcgill.ca>
"""

from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pickle
import datetime
import collections
from enum import Enum
from utils import *


class MLP(nn.Module):
	"""Multilayer perceptron"""

	def __init__(self):
		super(MLP, self).__init__()
		self.n_dropouts = 0

		# Layer 1
		self.fc_1 = nn.Linear(784, 800)
		self.relu_1 = nn.ReLU()

		# Layer 2
		self.fc_2 = nn.Linear(800, 800)
		self.relu_2 = nn.ReLU()

		# Output layer
		self.fc_3 = nn.Linear(800, 10)
		self.dropout = nn.Dropout(p=0.5)
		self.softmax = nn.Softmax(dim=0)

	def forward(self, x, n_dropouts=False, version=1):
		# Sample dropout for last layer
		out = self.fc_1(x)
		out = self.relu_1(out)
		out = self.fc_2(out)
		out = self.relu_2(out)
		out = self.fc_3(out)

		if n_dropouts:
			
			# Only do dropout in evaluation mode
			if self.training == False:
				masks = [self.dropout(out) for i in range(n_dropouts)]

				# Part (b) ii. Average dropouts before applying softmax
				if version == 2:
					out = sum(m for m in masks) / n_dropouts
					out = self.softmax(out)

				# Part (b) iii. Average predictions after applying softmax
				elif version == 3:
					out = sum(self.softmax(m) for m in masks) / n_dropouts

		# No dropout
		else:
			out = self.dropout(out)
			out = self.softmax(out)
		

		return out


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
		self.dense = nn.Linear(128, 10)

	def forward(self, x):
		return self.dense(self.model(x).squeeze())


class MNIST():
	"""Deep model for Problem 1"""

	def __init__(self, learning_rate, model_type, weight_decay=0, batch_norm=False):
		"""Initialize deep net"""

		self.learning_rate = learning_rate
		self.model_type = model_type
		self.weight_decay = weight_decay
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
			self.model = MLP()

		# Initialize weights
		self.model.apply(self.init_weights)

		# Set loss function and gradient-descend optimizer
		#if not self.dropout_masks or (self.dropout_masks and self.avg_pre_softmax):
		#	self.loss_fn = nn.CrossEntropyLoss()
		#else:
		#	self.loss_fn = nn.NLLLoss()
		self.loss_fn = nn.CrossEntropyLoss()

		# L2 regularization
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


	def predict_dropouts(self, data_loader):
		"""Evaluate model on dataset"""

		# Set model phase
		self.model.train(False)

		correct = np.zeros(len(self.dropout_masks))
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
			for j, n_masks in enumerate(self.dropout_masks):
				self.model.n_dropouts = n_masks
				y_pred = self.model(x)
				c = float((y_pred.max(1)[1] == y).sum().data[0]) / data_loader.batch_size
				correct[j] += c

		# Compute accuracy for each number of masks
		acc = [c / len(data_loader) for c in correct]
		return acc


	def predict(self, data_loader, n_dropout=False, version=1):

		# Set model phase
		self.model.train(False)

		correct = 0.
		for batch_idx, (x, y) in enumerate(data_loader):
			x, y = Variable(x).view(len(x), -1), Variable(y)

			if torch.cuda.is_available():
				x = x.cuda()
				y = y.cuda()

			# Predict
			y_pred = self.model(x, n_dropout, version)
			correct += float((y_pred.max(1)[1] == y).sum().data[0]) \
						/ data_loader.batch_size

		# Compute accuracy
		acc = correct / len(data_loader)
		return acc


	def train(self, nb_epochs, train_loader, valid_loader, test_loader):
		"""Train model on data"""

		# Format header and set phase
		#format_header(self)

		# Initialize tracked quantities
		train_loss, train_acc, valid_acc, test_acc, l2_norm = [], [], [], [], []

		# Train
		start = datetime.datetime.now()
		for epoch in range(nb_epochs):
			self.model.train(True)

			print("Epoch {:d} | {:d}".format(epoch + 1, nb_epochs))
			losses = AverageMeter()

			# Mini-batch SGD
			for batch_idx, (x, y) in enumerate(train_loader):
				# Print progress bar
				progress_bar(batch_idx, len(train_loader.dataset) / \
						train_loader.batch_size)

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

				# Get L2 norm of all parameters
				# params = [l.view(1,-1) for l in self.model.parameters()]
				# l2_norm.append(torch.cat(params, dim=1).norm().data[0])

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
			show_learning_stats(track,train_loss[epoch], np.mean(train_acc[epoch]),
					np.mean(valid_acc[epoch]), np.mean(test_acc[epoch]))

		# Print elapsed time
		end = datetime.datetime.now()
		elapsed = str(end - start)[:-7]
		print("Training done! Elapsed time: {}\n".format(elapsed))

		return l2_norm, train_loss, train_acc, valid_acc, test_acc


if __name__ == "__main__":


	# Enum to choose which model to train
	class Net(Enum):
		MLP = 0
		CNN = 1

	# Model parameters
	learning_rate = 0.02
	lmbda = 0
	batch_size = 64
	nb_epochs = 100
	model_type = Net.MLP
	dropout_masks = range(10,110,10)
	batch_norm = False
	data_filename = "../data/mnist/mnist.pkl"

	# Load data
	train_loader, valid_loader, test_loader = load_mnist(data_filename, batch_size)

	# Adjust weight decay for SGD mini-batch
	weight_decay = lmbda * batch_size / len(train_loader.dataset)

	net = MNIST(learning_rate, model_type, weight_decay, batch_norm)
	_, _, train_acc, _, test_acc = net.train(nb_epochs, train_loader, valid_loader, test_loader)

	v2, v3 = [], []
	v1 = net.predict(test_loader)
	v1 = [v1 for _ in range(10)]
	for n_dropout in dropout_masks:
		v2.append(net.predict(test_loader, n_dropout, 2))
		v3.append(net.predict(test_loader, n_dropout, 3))	

	data = [v1,v2,v3]
	with open('1b.pkl', 'wb') as fp:
		pickle.dump(data, fp)

	
	#with open('1b.pkl', 'rb') as fp:
	#	data = pickle.load(open('1b.pkl', 'rb'))
	#plots_per_epoch(data, ["No Dropout", "Pre-Softmax Avg", "Post-Softmax Avg"], "Accuracy", "Accuracy on Test Set for Different Dropout Schemes")



