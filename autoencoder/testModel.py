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

gan = Pix2Pix('',0 )

gan.combined.load_weights("bestWeights/ganWeights.h5")
gan.generator.load_weights("bestWeights/generatorWeights.h5")
gan.discriminator.load_weights("bestWeights/discriminatorWeights.h5")


image = cv2.imread('./project_data/classificationProcessed/val/Pleural_Thickening/00002123_000.png')/127.5 -1
print(image.shape)
gray = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2GRAY)
fake_A = gan.generator.predict(np.expand_dims(gray, axis=0))
gen_output_img = (fake_A +1) * 127.5
print(np.squeeze(gen_output_img).shape)
cv2.imwrite("test_example.png", np.squeeze(gray))
cv2.imwrite("test.png", np.squeeze(gen_output_img))
quit()
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
        cv2.imwrite("/home/nasheath_ahmed/X-RayShadowRemovalAndClassification/validated_images_OG_continued/" + imgpaths[im].split("/")[-1], combined_out_img)
        break

