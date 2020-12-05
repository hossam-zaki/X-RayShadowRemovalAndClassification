import argparse
import os
from scantree import scantree, RecursionFilter
import shutil
import math
import numpy as np
from tqdm import tqdm
import cv2
'''
This script will set up the data into respective train validation and test folders so that we can run the training algorithm on it. It takes in arguments
from the command line and there are defaults set to it.
'''
parser = argparse.ArgumentParser()
parser.add_argument('--datasetname', type=str, default='bone_supression_data', help="name of the directory of the data")
parser.add_argument('--basepath',type=str, default="/home/nasheath_ahmed/X-RayShadowRemovalAndClassification/project_data/")
parser.add_argument('--datastorage',type=str, default="/home/nasheath_ahmed/data/augmented/augmented/")
parser.add_argument('--testsplit', type=float, default=0.2, help='the test split as decimal')
parser.add_argument('--validationsplit', type=float, default=0.15, help='the validation split as a decimal')
opt = parser.parse_args()
#####Make the proper directories to store the data in######
if not os.path.exists(opt.basepath+opt.datasetname):
    os.mkdir(os.path.join(opt.basepath+opt.datasetname))
    os.mkdir(os.path.join(opt.basepath+opt.datasetname+'/train'))
    os.mkdir(os.path.join(opt.basepath+opt.datasetname+'/test'))
    os.mkdir(os.path.join(opt.basepath+opt.datasetname+'/val'))



##### Scan tree recursion to get all the image paths######
tree_source = scantree(opt.datastorage +'source/', RecursionFilter(match=['*.png']))
source_images =  [path.real for path in tree_source.filepaths()]
tree_target = scantree(opt.datastorage+'target/', RecursionFilter(match=['*.png']))
target_images =  [path.real for path in tree_target.filepaths()]




#######Same number of files in both#########

number_test_samples = math.ceil(len(source_images)*opt.testsplit)


for i in tqdm(range(3950,len(source_images))):
    source_image = cv2.imread(source_images[i])
    target_image = cv2.imread(target_images[i])
    concat_image = np.concatenate((source_image, target_image), 1)
    if(i== 3952):
        print(source_images[i])
        quit()
    # if i < number_test_samples:
    #     cv2.imwrite(os.path.join(opt.basepath+opt.datasetname+'/test/'+'combined_img_'+str(i)+'.png'), concat_image)
    # else:
    #     cv2.imwrite(os.path.join(opt.basepath+opt.datasetname+'/train/'+'combined_img_'+str(i)+'.png'), concat_image)
    




