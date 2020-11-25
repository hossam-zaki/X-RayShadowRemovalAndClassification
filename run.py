import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# from your_model import YourModel
# import hyperparameters as hp
from classificationData import Datasets
# from tensorboard_utils import ImageLabelingLogger, ConfusionMatrixLogger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def parse_args():
    """ Perform command-line argument parsing. """

    parser = argparse.ArgumentParser(
        description="Let's train some neural nets!")
    parser.add_argument(
        '--load-checkpoint',
        default=None,
        help='''Path to model checkpoint file (should end with the
        extension .h5). Checkpoints are automatically saved when you
        train your model. If you want to continue training from where
        you left off, this is how you would load your weights. In
        the case of task 2, passing a checkpoint path will disable
        the loading of VGG weights.''')
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='''Skips training and evaluates on the test set once.
        You can use this to test an already trained model by loading
        its checkpoint.''')

    return parser.parse_args()

def train(model, datasets, checkpoint_path):
    """ Training routine. """

    # Keras callbacks for training
    callback_list = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path + \
                    "weights.e{epoch:02d}-" + \
                    "acc{val_sparse_categorical_accuracy:.4f}.h5",
            monitor='val_sparse_categorical_accuracy',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            patience=2, 
            mode='min')
    ]


    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=10,
        batch_size=None,
        callbacks=callback_list,
    )

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=1,
    )


def main():
    """ Main function. """

    datasets = Datasets('project_data/classificationProcessed')

    convolutionBase = tf.keras.applications.ResNet50(
        include_top=False, 
        weights='imagenet', 
        input_shape=(1024, 1024, 3), 
        pooling=None)

    # convolutionBase.trainable = True
    # set_trainable = False
    # for layer in convolutionBase.layers:
    #     print(layer.name)
    #     if layer.name == 'conv5_block1_1_conv':
    #         set_trainable = True
    #         layer.trainable = set_trainable
    #     else:
    #         layer.trainable = set_trainable

    model = tf.keras.Sequential(
        [
            convolutionBase
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(1024, activation="relu", name="layer1"),
            # tf.keras.layers.Dense(512, activation="relu", name="layer2"),
            # tf.keras.layers.Dense(128, activation="relu", name="layer3"),
            # tf.keras.layers.Dense(9, name="layer4")
        ]
    )
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.Dense(1024, activation="relu"))
    # model.add(keras.layers.Dense(512, activation="relu"))
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(9, activation='softmax'))


    model(tf.keras.Input(shape=(1024, 1024, 3)))
    checkpoint_path = "./classificationWeights/"
    print(model.summary())

    # if ARGS.load_checkpoint is not None:
    #     model.load_weights(ARGS.load_checkpoint)

    # Compile model graph
    model.compile(
        optimizer='adam',
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"])

    if ARGS.evaluate:
        test(model, datasets.test_data)
    else:
        train(model, datasets, checkpoint_path)
        convolutionBase.trainable = True
        train(model, datasets, checkpoint_path)

# Make arguments global
ARGS = parse_args()

main()