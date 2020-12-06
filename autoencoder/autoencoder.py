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



# class Encoder(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.layer1 = tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer2 = tf.keras.layers.Conv2D(32, kernel_size=5, strides=2, padding='same',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer3 = tf.keras.layers.Conv2D(64, kernel_size=5, strides=2, padding='same',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer4 = tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer5 = tf.keras.layers.Conv2D(128, kernel_size=5, strides=2, padding='same',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer6 = tf.keras.layers.Conv2D(256, kernel_size=5, strides=2, padding='same',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#     # @tf.function
#     # def conv2d(layer_input, filters, f_size=4, bn=True):
#     #     d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
#     #     d = LeakyReLU(alpha=0.2)(d)
#     #     if bn:
#     #         d = BatchNormalization(momentum=0.8)(d)
#     #     return d


#     def call(self, images):
#         gf = 32
#         images = self.layer1(images)
#         images = LeakyReLU(alpha=0.2)(images)
#         images = BatchNormalization(momentum=0.8)(images)
#         images = self.layer2(images)
#         images = LeakyReLU(alpha=0.2)(images)
#         images = self.layer3(images)
#         images = LeakyReLU(alpha=0.2)(images)
#         images = self.layer4(images)
#         images = LeakyReLU(alpha=0.2)(images)
#         images = self.layer5(images)
#         images = LeakyReLU(alpha=0.2)(images)
#         images = self.layer6(images)
#         images = LeakyReLU(alpha=0.2)(images)

#         return images

# def deconv2d(layer_input, filters, f_size=4, dropout_rate=0):
#         """Layers used during upsampling"""
#         u = UpSampling2D(size=2)(layer_input)
#         u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
#         if dropout_rate:
#             u = Dropout(dropout_rate)(u)
#         u = BatchNormalization(momentum=0.8)(u)
#         return u
# class Decoder(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.layer1 = tf.keras.layers.Conv2D(256, kernel_size=5, strides=1, padding='same', activation='relu',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer2 = tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer3 = tf.keras.layers.Conv2D(128, kernel_size=5, strides=1, padding='same', activation='relu',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer4 = tf.keras.layers.Conv2D(64, kernel_size=5, strides=1, padding='same', activation='relu',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer5 = tf.keras.layers.Conv2D(32, kernel_size=5, strides=1, padding='same', activation='relu',kernel_initializer =tf.keras.initializers.RandomNormal(
#     mean=0.0, stddev=0.1, seed=None
# ))
#         self.layer6 = tf.keras.layers.Conv2D(3, kernel_size=5, strides=1, padding='same', activation='tanh')

#     def call(self, encoder_output):
#         print(encoder_output.shape)
#         image = tf.keras.layers.UpSampling2D(size=2)(encoder_output)
#         image = self.layer1(image)
#         image = BatchNormalization(momentum=0.8)(image)
#         image = UpSampling2D(size=2)(image)
#         image = self.layer2(image)
#         image = BatchNormalization(momentum=0.8)(image)
#         image = UpSampling2D(size=2)(image)
#         image = self.layer3(image)
#         image = BatchNormalization(momentum=0.8)(image)
#         image = UpSampling2D(size=2)(image)
#         image = self.layer4(image)
#         image = BatchNormalization(momentum=0.8)(image)
#         image = UpSampling2D(size=2)(image)
#         image = self.layer5(image)
#         image = BatchNormalization(momentum=0.8)(image)
#         image = UpSampling2D(size=2)(image)
#         output_img = self.layer6(image)
#         return output_img

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
       super(Encoder, self).__init__()
       self.encoder_conv_1 = tf.keras.layers.Conv2D(16,3,padding='same', strides= (2,2), kernel_initializer =tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.2, seed=None))
       self.encoder_conv_2 = tf.keras.layers.Conv2D(32,3,padding='same', strides= (2,2), kernel_initializer =tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.2, seed=None))
       self.encoder_conv_3 = tf.keras.layers.Conv2D(64,3,padding='same', strides= (2,2), kernel_initializer =tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.2, seed=None))

    @tf.function
    def call(self, images):
      layer1 = self.encoder_conv_1(images)
      layer1= tf.nn.leaky_relu(layer1, alpha=0.2)

      layer2 = self.encoder_conv_2(layer1)
      layer2= tf.nn.leaky_relu(layer2, alpha=0.2)

      layer3 = self.encoder_conv_3(layer2)
      layer3= tf.nn.leaky_relu(layer3, alpha=0.2)
      return layer3

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.decoder_deconv_1 = tf.keras.layers.Conv2D(32,3,padding='same', strides= (1,1), kernel_initializer =tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.2, seed=None))

        self.decoder_deconv_2 = tf.keras.layers.Conv2D(16,3,padding='same', strides= (1,1), kernel_initializer =tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.2, seed=None))
        self.decoder_deconv_3 = tf.keras.layers.Conv2D(3,3,padding='same', strides= (1,1), kernel_initializer =tf.keras.initializers.RandomNormal(
    mean=0.0, stddev=0.2, seed=None))
        
    @tf.function
    def call(self, encoder_output):

        data = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(encoder_output)
        data = self.decoder_deconv_1(data)
        data = tf.nn.leaky_relu(data, alpha=0.2)

        data = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(data)
        data = self.decoder_deconv_2(data)
        data = tf.nn.leaky_relu(data, alpha=0.2)

        data = tf.keras.layers.UpSampling2D(size=(2,2), interpolation='nearest')(data)
        data = self.decoder_deconv_3(data)
        results = tf.nn.leaky_relu(data, alpha=0.2)
        return results

class Autoencoder(tf.keras.Model):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def call(self, images):
        encoder = self.encoder(images)
        decoded = self.decoder(encoder)
        return decoded
    
    
    def loss_function(self, encoded, originals):
        mse = tf.reduce_mean(tf.reduce_mean(tf.math.squared_difference(encoded, originals), 1))
        ssim = tf.reduce_mean(1 - tf.image.ssim_multiscale(tf.cast(encoded, tf.float32), tf.cast(originals, tf.float32), 1))
        return .85*ssim + (1 - .85)*mse

def train(model, optimizer, source_images, target_images, totalLoss):
    with tf.GradientTape() as tape:
        output = model.call(source_images)
        loss = model.loss_function(output, target_images)
        print(loss)
        totalLoss.append(loss)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return output, totalLoss










if __name__ == '__main__':
    # parser = ArgumentParser()

    # parser.add_argument('--batch',type=int, default=4, required=False)
    # parser.add_argument('--epochs',type=int, default=25, required=False)
    # parser.add_argument('--loadWeights',type=bool, default=False, required=False)
    # parser.add_argument('--pathToWeights',type=str, default='', required=False)
    # parser.add_argument('--pathToData',type=str, default='project_data/bone_suppression_data/', required=True)
    # parser.add_argument('--lr', type=float, default=.001, required=False)
    # config = parser.parse_args()

    # gan = Pix2Pix(config.pathToData, config.lr)
    # if config.loadWeights:
    #     print("hia")
    #     gan.combined.load_weights(f"{config.pathToWeights}/ganWeights.h5")
    #     gan.generator.load_weights(f"{config.pathToWeights}/generatorWeights.h5")
    #     gan.discriminator.load_weights(f"{config.pathToWeights}/discriminatorWeights.h5")
    # gan.train(epochs=config.epochs, batch_size=config.batch, sample_interval=200)
    # gan.combined.save_weights("ganWeights.h5")
    # gan.generator.save_weights("generatorWeights.h5")
    # gan.discriminator.save_weights("discriminatorWeights.h5")
    data_loader = DataLoader(dataset_name='project_data/bone_supression_data',
                                      img_res=(1024, 1024))
    model = Autoencoder()
    model.built = True
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001) #initial learning rate is .001
    epochs = 50
    batch_size = 4
    train(model, optimizer, np.zeros((1, 1024, 1024, 3)), np.zeros((1, 1024, 1024, 3)), [])
    model.load_weights('autoencoderWeights/23_27.40587615966797_autoencoderWeights.h5')

    with open('trainingAutoencoder.txt', 'w+') as f:
        for epoch in range(epochs):
            f.write(f"Epoch {epoch}/{epochs}")
            if epoch + 1 % 25 == 0:
                print("in")
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            totalLoss = []
            for batch_i, (imgs_A, imgs_B, imgpaths) in enumerate(data_loader.load_batch(batch_size)):

                fake_A, totalLoss = train(model, optimizer, imgs_B, imgs_A, totalLoss)
                if batch_i % 50 == 0:
                    for im in range(len(fake_A)):
                        gen_output_img = (fake_A[im] + 1) * 127.5
                        org_data = cv2.imread(imgpaths[im])
                        org_data = cv2.resize(org_data, (1024 *2, 1024))
                        org_data_left = org_data[:, :1024,:]
                        org_data_right = org_data[:, 1024:,:]
                        combined_out_img = np.concatenate((org_data_left,org_data_right, gen_output_img), 1)
                        print(imgpaths[im])
                        cv2.imwrite("/home/nasheath_ahmed/X-RayShadowRemovalAndClassification/autoencoderContinued/" + str(epoch) + "_" + str(batch_i) + "_" + imgpaths[im].split("/")[-1], combined_out_img)
                        break
            total_loss = np.sum(totalLoss)
            print(f"Loss of epoch: {totalLoss}")
            model.save_weights(f'autoencoderWeights/{epoch + 25}_{total_loss}_autoencoderWeights.h5')
            f.write(f"Epoch {epoch + 25} loss : {total_loss}")
    
    
    
    receiver_emails = ['hossam_zaki@brown.edu', 'mohamad_abouelafia@brown.edu', 'nasheath_ahmed@brown.edu', 'andrew_aoun@brown.edu']
    for bozo in receiver_emails:
        port = 465
        smtp_server = "smtp.gmail.com"
        sender_email = "nasheathpython@gmail.com" # Enter your address
        receiver_email = bozo # Enter receiver address
        password = "nasheath24!"
        message = 'Subject: {}\n\n{}'.format("whaddup bimbos", "This message is sent from Python saying your training is finished! aHaHa.")

        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message)

