import os
import glob
import pickle
import pandas as pd
import numpy as np
from skimage import io
from skimage.transform import resize


class ImagePreprocessor():
    def __init__(self, src_dir, dest_dir):
        self.src_dir = src_dir
        self.dest_dir = dest_dir


    def get_file_list(self):
        f = os.listdir(path=self.src_dir)
        file_list = sorted(f)

        return file_list


    def resize_save(self, save=False, color=False):
        '''
        The image transform pipeline. Will resize, grayscale, transform, save individual images, and pickle
        the image stream.

        Input:

            src_path - a string with the absolute path of the raw images
            dest_path - a string with the absolute path of the processed images
            save - to save or not to save each individual transformed image. default = False
            color - saves a grayscale image if False, color image if True. 
            default = False

        Output:

            returns - the image_set 
        '''
        if save:
            for file in glob.glob(self.src_dir + '/*'):
                f_name = file.split('/')[-1].split('.')[0]
                name = self.dest_dir + '/' + f_name + '.jpg'
                if color:
                    resized = resize(io.imread(file), (128, 128), anti_aliasing=True)
                    # colour = pd.Series(resized, name=file)
                    io.imsave(name, resized)
                if not color:
                    resized = resize(io.imread(file), (128, 128), anti_aliasing=True)
                    gray = pd.Series(color.rgb2gray(resized), name=file)
                    io.imsave(os.path.join(self.dest_dir, name), gray)

        else: 
            image_set = np.ndarray()

            for file in glob.glob(self.src_dir + '/*'):
                resized = resize(io.imread(file), (128, 128), anti_aliasing=True)
                
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
    src_dir = 'data/raw/images'
    dest_dir = 'data/processed/images'
    pickle_dir = 'data/processed/'

    image_pre = ImagePreprocessor(src_dir, dest_dir)
    
    image_set = image_pre.resize_save(save=False, color=True)

    file_list = image_pre.get_file_list()

