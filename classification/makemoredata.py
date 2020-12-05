import tensorflow as tf 
import numpy as np
import os
from scantree import scantree, RecursionFilter
import cv2
import imutils 


tree_source = scantree('./project_data/classificationProcessed/val', RecursionFilter(match=['*.png']))
all_images =  [path.real for path in tree_source.filepaths()]
tree_source = scantree('./project_data/classificationProcessed/train/Nodule', RecursionFilter(match=['*.png']))
nods =  [path.real for path in tree_source.filepaths()]
tree_source = scantree('./project_data/classificationProcessed/train/Pleural_Thickening', RecursionFilter(match=['*.png']))
plt =  [path.real for path in tree_source.filepaths()]
tree_source = scantree('./project_data/classificationProcessed/train/Pneumothorax', RecursionFilter(match=['*.png']))
pneum =  [path.real for path in tree_source.filepaths()]
all_images = all_images + nods + plt + pneum

base_folder = './project_data/classificationProcessed/'
for image_ in all_images:
    base_image_name = os.path.basename(image_)
    split_arr = image_.split('/')
    type_of_folder = split_arr[len(split_arr)-3]
    # image = cv2.imread(image_)

    image = cv2.imread(image_)
    image = imutils.rotate(image, np.random.randint(15, size=1)[0])
    cv2.imwrite('./project_data/classificationProcessed/'+type_of_folder +'/'+ split_arr[len(split_arr)-2] + '/' + "rotate" + base_image_name, np.squeeze(image))

    image = cv2.imread(image_)
    image = np.fliplr(image)
    cv2.imwrite('./project_data/classificationProcessed/'+type_of_folder +'/'+ split_arr[len(split_arr)-2] + '/' + "flip" + base_image_name, np.squeeze(image))
    print('./project_data/classificationProcessed/'+type_of_folder +'/'+ split_arr[len(split_arr)-2] + '/' + "flip" + base_image_name)
    #quit()
