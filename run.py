import os
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.metrics import AUC

from model import ChestClassModel
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
                    "acc{val_acc:.4f}.h5",
            monitor='val_acc',
            save_best_only=True,
            save_weights_only=True),
        tf.keras.callbacks.TensorBoard(
            update_freq='batch',
            profile_batch=0),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='loss', 
            patience=2, 
            mode='min')
    ]


    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=50,
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

    model = ChestClassModel()
    model(tf.keras.Input(shape=(512, 512, 3)))
    checkpoint_path = "./classificationWeights4/"

    print(model.summary())

    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint)

    # Compile model graph
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    #changed from sparse_categorical 
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[AUC(multi_label=True), "acc", "binary_accuracy"])

    if ARGS.evaluate:
        test(model, datasets.test_data)
    else:
        train(model, datasets, checkpoint_path)
    model.save_weights('./classificationWeights4/allWeights.h5')
# Make arguments global
ARGS = parse_args()

main()