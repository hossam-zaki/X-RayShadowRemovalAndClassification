# X-RayShadowRemovalAndClassification

    --> Autoencoder

No known bugs. The autoencoder folder deals with all code related to suppressing bone shadows.

The pix2pix.py file has code related to the GAN model that was initially implemented. The current iteration uses the autoencoder.py file for this feature suppression.

The preprocess .py file will set up the data into respective train validation and test folders so that we can run the training algorithm on it. It takes in arguments, variable from the command line, and there are defaults set to it. These include:

    --datasetname (input data name)
    --basepath (project path)
    --datastorage (data folder path)

The data loader file will deal with loading in the data to preprocess

image.py will run all validation images on a pre-loaded model

    --> Classification

No known bugs. The classification folder deals with all code related to classifiying disease images both with and without feature suppression. All relevant files will be explained in this section.

The classificationData.py file provides functions for pre-processing and preparing the data.

makemoredata.py is a file used to make rotated copies of all images in the classification set. In addition, the images will be flipped and a copy of the flipped image will also be made tripling the data set size.

Model.py builds on the Resnet architecture pre-trained with Imagenet

The run.py file allows the user to define arguments and run the classification model. Arguments include:

    --datasetname (input data path)
    --epoch
    --batchsize
    --verbose
    --patience
    --learning rate

In addition, the classification folder provides two jupyter notebooks with code on generating ROC curves and preprocessing the classification data giving the user a more specific walkthrough of these processes. The generate ROC curves notebooks, all that needs to be done is plugging in the relevant model weights as well as the path to the validation images. For the preprocessing notebook, the user can input the path for the location of saved figures.
