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
from scantree import scantree, RecursionFilter


from pix2pix import Pix2Pix

gan = Pix2Pix('',0 )

gan.combined.load_weights("bestWeights/ganWeights.h5")
gan.generator.load_weights("bestWeights/generatorWeights.h5")
gan.discriminator.load_weights("bestWeights/discriminatorWeights.h5")
tree_source = scantree('./project_data/classificationProcessed/', RecursionFilter(match=['*.png']))
all_images =  [path.real for path in tree_source.filepaths()]

#brightens initial image
if not os.path.exists('./project_data/classification_suppressed'):
    os.mkdir('./project_data/classification_suppressed')
    os.mkdir('./project_data/classification_suppressed/train')
    os.mkdir('./project_data/classification_suppressed/test')
    os.mkdir('./project_data/classification_suppressed/val')


base_folder = './project_data/classification_suppressed/'
for i in ['test', 'train', 'val']:
    for image_ in all_images:
        base_image_name = os.path.basename(image_)
        split_arr = image_.split('/')
        type_of_folder = split_arr[len(split_arr)-3]
        if not os.path.exists('./project_data/classification_suppressed/'+type_of_folder +'/'+ split_arr[len(split_arr)-2]):
            os.mkdir('./project_data/classification_suppressed/'+ type_of_folder+'/'+ split_arr[len(split_arr)-2])

        image = cv2.imread(image_)
        original = image.copy()
        indices = image > 140
        image[indices] = image[indices] + 30

        #inverts the image so the bone shadowed area is not in a  lighter shade
        inverted = cv2.bitwise_not(image)
        #reduce the brightness on the area where the bone shadows are. 
        indices = inverted > 140
        inverted[indices] = inverted[indices]/ 1.5

        inverted = inverted/127.5 - 1
        # test = original/127.5 -1 
        fake_A = gan.generator.predict(np.expand_dims(inverted, axis=0))

        gen_output_img = (fake_A +1) * 127.5
        cv2.imwrite('./project_data/classification_suppressed/'+type_of_folder +'/'+ split_arr[len(split_arr)-2] + '/' + base_image_name, np.squeeze(gen_output_img))
