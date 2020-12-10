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
    parser.add_argument('--datasetname', type=str, default="/home/nasheath_ahmed/X-RayShadowRemovalAndClassification/project_data/classificationProcessed", help="name of the directory of the data")
    parser.add_argument('--epochs', type=float, default=25, help='the number of epochs to train for')
    parser.add_argument('--batchsize', type=float, default=4, help='the batch size to use')
    parser.add_argument('--patience', type=float, default=1, help='the patience for LR on plateau')
    parser.add_argument('--verbose', type=float, default=2, help='the verbose setting on test')
    parser.add_argument('--learningrate', type=float, default=.00001, help='the optimizer learning rate')

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
            patience=ARGS.patience, 
            mode='min')
    ]


    # Begin training
    model.fit(
        x=datasets.train_data,
        validation_data=datasets.test_data,
        epochs=ARGS.epochs,
        batch_size=ARGS.batchsize,
        callbacks=callback_list,
    )

def test(model, test_data):
    """ Testing routine. """

    # Run model on test set
    model.evaluate(
        x=test_data,
        verbose=ARGS.verbose,
    )


def main():
    """ Main function. """

    # datasets = Datasets('../project_data/classificationProcessed')
    datasets = Datasets(ARGS.basepath)

    model = ChestClassModel()
    model(tf.keras.Input(shape=(512, 512, 3)))
    checkpoint_path = "./classificationWeights5/"

    print(model.summary())

    if ARGS.load_checkpoint is not None:
        model.load_weights(ARGS.load_checkpoint)

    # Compile model graph
    # optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    optimizer = tf.keras.optimizers.Adam(learning_rate=ARGS.learningrate)
    #changed from sparse_categorical 
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[AUC(multi_label=True), "acc", "binary_accuracy"])

    if ARGS.evaluate:
        test(model, datasets.test_data)
    else:
        train(model, datasets, checkpoint_path)
    model.save_weights('./classificationWeights5/allWeights.h5')
# Make arguments global
ARGS = parse_args()

main()