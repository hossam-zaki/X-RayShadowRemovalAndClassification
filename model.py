import tensorflow as tf
# import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense

class ChestClassModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(ChestClassModel, self).__init__()

        convolutionBase = tf.keras.applications.ResNet50(
            include_top=False, 
            weights='imagenet', 
            input_shape=(512, 512, 3), 
            pooling=None)

        self.architecture = [
            convolutionBase,
            Flatten(),
            Dense(512, activation='relu'),
            Dropout(.5),
            Dense(9, activation='softmax'),
        ]

        # ====================================================================

    def call(self, img):
        """ Passes input image through the network. """

        for layer in self.architecture:
            img = layer(img)

        return img

    # @staticmethod
    # def loss_fn(labels, predictions):
    #     """ Loss function for the model. """

    #     return tf.keras.losses.sparse_categorical_crossentropy(
    #         labels, predictions, from_logits=False)