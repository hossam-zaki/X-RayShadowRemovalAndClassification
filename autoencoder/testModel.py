from __future__ import print_function, division
import scipy
import tensorflow as tf
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
from dataloader import DataLoader
import numpy as np
import os
import cv2
import smtplib, ssl
import argparse
from argparse import ArgumentParser
import cv2

from autoencoder import Autoencoder

image = cv2.imread('project_data/classificationProcessed/test/Atelectasis/00001088_017.png')

autoencoder = Autoencoder()

autoencoder.predict(tf.cast(tf.expand_dims(image, axis=0), tf.float32))
autoencoder.load_weights('autoencoderWeights/23_27.40587615966797_autoencoderWeights.h5')

prediction = autoencoder.predict(tf.cast(tf.expand_dims(image, axis=0), tf.float32))

hmchong = np.float32(cv2.imread('../ML-BoneSuppression/tester.png'))
ours = np.float32(cv2.imread('tester.png'))

mse = tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(hmchong, ours), 1))
ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(tf.cast(hmchong, tf.float32), tf.cast(ours, tf.float32), 1))
print(.85*ssim + (1 - .85)*mse)

cv2.imwrite('tester.png', np.float32(tf.squeeze(prediction)))
