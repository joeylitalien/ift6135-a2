#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load Cats & Dogs Kaggle dataset
Assumes train.zip has been unzipped in savedir

@author: chinwei, joeylitalien
"""

from __future__ import print_function

import urllib.request, urllib.parse
import pickle
import gzip
import os
import numpy as np
import zipfile
import scipy.ndimage


final_size = 64

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='datasets', 
                        help='directory to save the dataset')
    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    train_data = os.path.join(args.savedir, 'train')
    train_proc_data = os.path.join(args.savedir, 'train_64x64')
    if not os.path.exists(train_proc_data):
        os.makedirs(train_proc_data)

    for pic_file in os.listdir(train_data):
        pic_path = os.path.join(train_data, pic_file)
        img = scipy.ndimage.imread(pic_path)
        side_dim = min(img.shape[0], img.shape[1])
        start_height = (img.shape[0] - side_dim) // 2
        start_width = (img.shape[1] - side_dim) // 2
        img = img[start_height: start_height + side_dim,
                  start_width: start_width + side_dim]
        img = scipy.misc.imresize(
            img,
            size=float(final_size) / img.shape[0],
            interp='bilinear'
        )

        if (img.shape[0] != final_size or
            img.shape[1] != final_size):
            img = img[:final_size, :final_size]

        scipy.misc.imsave(
            os.path.join(train_proc_data, pic_file),
            img
        )

