from __future__ import print_function, division
import scipy
import tensorflow as tf
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, Conv2DTranspose
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

#notify me when of training progress

class Pix2Pix():
    def __init__(self):
        # Input shape
        self.img_rows = 1024
        self.img_cols = 1024
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        # Configure data loader
        self.dataset_name = 'bone_supression_data'
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))
        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)
        # Number of filters in the first layer of G and D
        self.gf = 32
        self.df = 32
        self.encoder = Encoder()
        self.decoder = Decoder()
        optimizer = Adam(0.001, 0.5)
        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        #Loading in the weights here NEED TO DELETE AFTER THE RUN WE DO  TODAY
        #self.discriminator.load_weights("discriminatorWeights.h5")
        self.discriminator.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------
        # Build the generator
        self.generator = self.build_generator()
        print(self.generator.summary())
        # Input images and their conditioning images
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        # By conditioning on B generate a fake version of A
        fake_A = self.generator(img_B)
        #Loading in the weights here NEED TO DELETE AFTER THE RUN WE DO  TODAY
        #self.generator.load_weights("generatorWeights.h5")
        # For the combined model we will only train the generator
        self.discriminator.trainable = False
        # Discriminators determines validity of translated images / condition pairs
        valid = self.discriminator([fake_A, img_B])

        self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])

        #Loading in the weights here NEED TO DELETE AFTER THE RUN WE DO  TODAY
        #self.combined.load_weights("ganWeights.h5")
        self.combined.compile(loss=['mse', 'mae'],
                              loss_weights=[1, 100],
                              optimizer=optimizer)

    def build_generator(self):
        """U-Net Generator"""

        inp = Input(shape=self.img_shape)
        data = self.encoder(inp)
        output_img = self.decoder(data)

        return Model(inp, output_img)

    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, bn=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)
        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])
        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)
        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        return Model([img_A, img_B], validity)
    def train(self, epochs, batch_size=8, sample_interval=50):
        start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)
        for epoch in range(epochs):
            for batch_i, (imgs_A, imgs_B, imgpaths) in enumerate(self.data_loader.load_batch(batch_size)):
                # ---------------------
                #  Train Discriminator
                # ---------------------
                # Condition on B and generate a translated version
                fake_A = self.generator.predict(imgs_B)
                if batch_i % 50 == 0:
                    for im in range(len(fake_A)):
                        gen_output_img = (fake_A[im] + 1) * 127.5
                        org_data = cv2.imread(imgpaths[im])
                        org_data = cv2.resize(org_data, (self.img_rows *2, self.img_cols))
                        org_data_left = org_data[:, :self.img_cols,:]
                        org_data_right = org_data[:, self.img_cols:,:]
                        combined_out_img = np.concatenate((org_data_left,org_data_right, gen_output_img), 1)
                        print(imgpaths[im])
                        cv2.imwrite("/home/nasheath_ahmed/X-RayShadowRemovalAndClassification/generated_images_4/" + str(epoch) + "_" + str(batch_i) + "_" + imgpaths[im].split("/")[-1], combined_out_img)
                        break
                # Train the discriminators (original images = real / generated = Fake)
                d_loss_real = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
                d_loss_fake = self.discriminator.train_on_batch([fake_A, imgs_B], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
                # -----------------
                #  Train Generator
                # -----------------
                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_A])
                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                with open("training1.txt", "a") as myfile:
                    myfile.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s \n" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time)) 
                print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                        batch_i, self.data_loader.n_batches,
                                                                        d_loss[0], 100*d_loss[1],
                                                                        g_loss[0],
                                                                        elapsed_time))
                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)
    def sample_images(self, epoch, batch_i):
        os.makedirs('sampled_images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3
        imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_testing=True)
        fake_A = self.generator.predict(imgs_B)
        gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        titles = ['Condition', 'Generated', 'Original']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[i])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("sampled_images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
       super(Encoder, self).__init__()
       self.encoder_conv_1 = tf.keras.layers.Conv2D(16, 5, strides=(1,1), padding='SAME', activation='relu', kernel_initializer=tf.initializers.random_normal(stddev=0.1))
       self.encoder_conv_2 = tf.keras.layers.Conv2D(32, 5, strides=(1,1), padding='SAME', activation='relu', kernel_initializer=tf.initializers.random_normal(stddev=0.1))
       self.encoder_conv_3 = tf.keras.layers.Conv2D(64, 5, strides=(1,1), padding='SAME', activation='relu', kernel_initializer=tf.initializers.random_normal(stddev=0.1))
    
    @tf.function
    def call(self, images):
        data = self.encoder_conv_1(images)
        data = tf.nn.max_pool(data, [1,2,2,1], [1,2,2,1], padding='SAME')
        data = self.encoder_conv_2(data)
        data = tf.nn.max_pool(data, [1,2,2,1], [1,2,2,1], padding='SAME')
        data = self.encoder_conv_3(data)
        data = tf.nn.max_pool(data, [1,2,2,1], [1,2,2,1], padding='SAME')
        return data

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.decoder_deconv_1 = tf.keras.layers.Conv2D(32, 5, strides=(1,1), 
            padding='SAME', activation='relu', kernel_initializer=tf.initializers.random_normal(stddev=0.02))
        self.decoder_deconv_2 = tf.keras.layers.Conv2D(16, 5, strides=(1,1), 
            padding='SAME', activation='relu', kernel_initializer=tf.initializers.random_normal(stddev=0.02))
        self.decoder_deconv_3 = tf.keras.layers.Conv2D(3, 5, strides=(1,1), 
            padding='SAME', activation='tanh', kernel_initializer=tf.initializers.random_normal(stddev=0.1))

    @tf.function
    def call(self, encoder_output):
        
        data = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(encoder_output)
        data = self.decoder_deconv_1(data)
        data = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(data)
        data = self.decoder_deconv_2(data)
        data = tf.keras.layers.UpSampling2D(size=2, interpolation='nearest')(data)
        data = self.decoder_deconv_3(data)
        return data

if __name__ == '__main__':
    gan = Pix2Pix()
    gan.train(epochs=50, batch_size=4, sample_interval=200)
    gan.combined.save_weights("weights_run_4/ganWeights.h5")
    gan.generator.save_weights("weights_run_4/generatorWeights.h5")
    gan.discriminator.save_weights("weights_run_4/discriminatorWeights.h5")
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