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


from pix2pix import Pix2Pix

gan = Pix2Pix()

gan.combined.load_weights("weight_run_3/ganWeights.h5")
gan.generator.load_weights("weight_run_3/generatorWeights.h5")
gan.discriminator.load_weights("weight_run_3/discriminatorWeights.h5")

# fake_A = gan.generator.predict(tf.expand_dims(cv2.imread('../data/augmented/augmented/source/0_1.png'), axis=0))
# cv2.imwrite("/home/nasheath_ahmed/X-RayShadowRemovalAndClassification/test.png", fake_A)
# quit()
data_loader = DataLoader(dataset_name="bone_supression_data",
                                      img_res=(1024, 1024))
for batch_i, (imgs_A, imgs_B, imgpaths) in enumerate(data_loader.load_batch(1, is_testing=True)):
    # ---------------------
    #  Train Discriminator
    # ---------------------
    # Condition on B and generate a translated version
    fake_A = gan.generator.predict(imgs_B)
    for im in range(len(fake_A)):
        gen_output_img = (fake_A[im] + 1) * 127.5
        org_data = cv2.imread(imgpaths[im])
        org_data = cv2.resize(org_data, (gan.img_rows *2, gan.img_cols))
        org_data_left = org_data[:, :gan.img_cols,:]
        org_data_right = org_data[:, gan.img_cols:,:]
        combined_out_img = np.concatenate((org_data_left,org_data_right, gen_output_img), 1)
        print(imgpaths[im])
        cv2.imwrite("/home/nasheath_ahmed/X-RayShadowRemovalAndClassification/validated_images_new_1/" + imgpaths[im].split("/")[-1], combined_out_img)
        break

