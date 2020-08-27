import os
import glob
import pickle
import pandas as pd
import numpy as np
from skimage import io, color, filters
from skimage.transform import resize, rotate


class ImagePreprocessor():
    def __init__(self, src_dir, dest_dir):
        self.src_dir = src_dir
        self.dest_dir = dest_dir


    def get_file_list(self):
        f = os.listdir(path=self.src_dir)
        file_list = sorted(f)

        return file_list


    def resize_save(self, save=False):
        '''
        The image transform pipeline. Will resize, grayscale, transform, save individual images, and pickle
        the image stream.

        Input:

            src_path - a string with the absolute path of the raw images
            dest_path - a string with the absolute path of the processed images
            save - to save or not to save each individual transformed image. Default = False
            pickle - to pickle the whole image set. Default = False

        Output:

            returns - the image_set 
        '''
        if save:
            for file in glob.glob(self.src_dir + '/*'):
                f_name = file.split('/')[-1].split('.')[0]
                name = self.dest_dir + '/' + f_name + '.jpg'
                resized = resize(io.imread(file), (128, 128), anti_aliasing=True)
                io.imsave(os.path.join(self.dest_dir, name), resized)

        else: 
            image_set = np.ndarray()

            for file in glob.glob(self.src_dir + '/*'):
                resized = resize(io.imread(file), (128, 128), anti_aliasing=True)
                gray = pd.series(color.rgb2gray(resized), name=file)
                image_set.append(np.asarray(gray))

            return image_set


    def pickle_it(self, lst, dir, filename):
        '''
        Pickles an object

        Input:

            lst - the object we want to pickle
            dir - a string with the directory we would like to save to
            filename - a string with the filename we want to use

        Output:

            a pickle file in the specified directory with the specified filename
        '''
        with open(os.path.join(dir, filename), 'wb') as fh:
            pickle.dump(lst, fh)

        pass


if __name__ == '__main__':
    src_dir = '/home/chris/Dropbox/galvanize/capstones/lemon-defect-detection/data/raw/images'
    dest_dir = '/home/chris/Dropbox/galvanize/capstones/lemon-defect-detection/data/processed/images'
    pickle_dir = '/home/chris/Dropbox/galvanize/capstones/lemon-defect-detection/data/processed/'

    image_pre = ImagePreprocessor(src_dir, dest_dir)
    
    image_set = image_pre.resize_save()

    file_list = image_pre.get_file_list()

