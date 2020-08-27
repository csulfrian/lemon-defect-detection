import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import sys
sys.path.append('./src')
from img_proc import ImagePreprocessor
from dataset_prep import AnnotationsParser, DatasetBuilder
from eda import MakePlots


if __name__ == '__main__':
    '''
    src_dir - the directory that contains the raw image files
    dest_dir - the directory that you want to save the converted images in
    ann_dir - the directory that contains the COCO-formatted annotations file
    fname - the file name of the annotations file
    '''
    src_dir = '/home/chris/Dropbox/galvanize/capstones/lemon-defect-detection/data/raw/images'
    dest_dir = '/home/chris/Dropbox/galvanize/capstones/lemon-defect-detection/data/processed/images'
    ann_dir = '/home/chris/Dropbox/galvanize/capstones/lemon-defect-detection/data/raw/annotations'
    fname = 'instances_default.json'

    image_pre = ImagePreprocessor(src_dir, dest_dir)
    image_pre.resize_save(save=False)

    lemons = DatasetBuilder(ann_dir, fname)
    X, y = lemons.load_data()
  
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
