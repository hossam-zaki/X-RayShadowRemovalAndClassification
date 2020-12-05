import os
import random
import numpy as np
from PIL import Image
import tensorflow as tf

class Datasets():
    """ Class for containing the training and test sets as well as
    other useful data-related information. Contains the functions
    for preprocessing.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        

        # For storing list of classes
        self.classes = [""] * 9

        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Setup data generators
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"),True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"),False, False)
        self.val_data = self.get_data(
            os.path.join(self.data_path, "val/"), False, False)

    
    def standardize(self, img):
        """ Function for applying standardization to an input image.
        Arguments:
            img - numpy array of shape (image size, image size, 3)
        Returns:
            img - numpy array of shape (image size, image size, 3)
        """

        img = (img - self.mean)/self.std 

        return img

    def preprocess_fn(self, img):
        """ Preprocess function for ImageDataGenerator. """

        img = img / 255.
        img = self.standardize(img)

        return img
    def get_data(self, path, shuffle, augment):
        """ Returns an image data generator which can be iterated
        through for images and corresponding class labels.
        Arguments:
            path - Filepath of the data being imported, such as
                   "../data/train" or "../data/test"
            shuffle - Boolean value indicating whether the data should
                      be randomly shuffled.
            augment - Boolean value indicating whether the data should
                      be augmented or not.
        Returns:
            An iterable image-batch generator
        """

        if augment:
            # TODO: Use the arguments of ImageDataGenerator()
            #       to augment the data. Leave the
            #       preprocessing_function argument as is unless
            #       you have written your own custom preprocessing
            #       function (see custom_preprocess_fn()).
            #
            # Documentation for ImageDataGenerator: https://bit.ly/2wN2EmK
            #
            # ============================================================

            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn,
                zoom_range = 0.2,)

            # ============================================================
        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
                preprocessing_function=self.preprocess_fn)

        # VGG must take images of size 224x224
        img_size = 1024

        classes_for_flow = None

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(512, 512),
            class_mode='categorical',
            batch_size=5,
            # color_mode= 'grayscale',
            shuffle=shuffle)

        return data_gen