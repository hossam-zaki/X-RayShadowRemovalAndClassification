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
        
        # Dictionaries for (label index) <--> (class name)
        self.idx_to_class = {}
        self.class_to_idx = {}

        # For storing list of classes
        self.classes = [""] * 9

        # Setup data generators
        self.train_data = self.get_data(
            os.path.join(self.data_path, "train/"),True, True)
        self.test_data = self.get_data(
            os.path.join(self.data_path, "test/"),False, False)
        self.val_data = self.get_data(
            os.path.join(self.data_path, "val/"), False, False)


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
                zoom_range = 0.2,
                horizontal_flip=True,
                featurewise_std_normalization=True)

            # ============================================================
        else:
            # Don't modify this
            data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
               featurewise_std_normalization=True)

        # VGG must take images of size 224x224
        img_size = 1024

        classes_for_flow = None

        # Make sure all data generators are aligned in label indices
        if bool(self.idx_to_class):
            classes_for_flow = self.classes

        # Form image data generator from directory structure
        data_gen = data_gen.flow_from_directory(
            path,
            target_size=(img_size, img_size),
            class_mode='sparse',
            batch_size=4,
            shuffle=shuffle,
            classes=classes_for_flow)

        # Setup the dictionaries if not already done
        if not bool(self.idx_to_class):
            unordered_classes = []
            for dir_name in os.listdir(path):
                if os.path.isdir(os.path.join(path, dir_name)):
                    unordered_classes.append(dir_name)

            for img_class in unordered_classes:
                self.idx_to_class[data_gen.class_indices[img_class]] = img_class
                self.class_to_idx[img_class] = int(data_gen.class_indices[img_class])
                self.classes[int(data_gen.class_indices[img_class])] = img_class

        return data_gen