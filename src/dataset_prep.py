import pandas as pd
import numpy as np
import glob
from skimage import io, color
from skimage.filters import sobel
from skimage.feature import canny
# from pycocotools.coco import COCO


class AnnotationsParser():
    def __init__(self, dir_, filename):
        self.dir_ = dir_
        self.filename = filename

    def _json_to_df(self):
        '''
        Creates dataframes from our COCO-formatted .json file

        Input:

            dir - str containing the directory path
            filename - str containing the filename to read

        Output:

            cats, anns, img - dataframes for each of the three relevant
            kinds of information in
                .json file
        '''
        coco = COCO(self.dir_ + '/' + self.filename)

        cats = coco.loadCats(coco.getCatIds())
        cats_df = pd.DataFrame.from_dict(cats)\
                              .set_index('id')\
                              .drop('supercategory', axis=1)

        anns = coco.loadAnns(coco.getAnnIds())
        anns_df = pd.DataFrame.from_dict(anns)\
                              .set_index('id')\
                              .drop('iscrowd', axis=1)
        anns_df.rename(columns={'image_id': 'image',
                                'category_id': 'category'}, inplace=True)
        anns_df.rename_axis('ann_num', inplace=True)

        catIds = coco.loadCats()
        imgIds = coco.getImgIds(catIds=catIds)
        img = coco.loadImgs(imgIds)
        to_del = ['date_captured',
                  'coco_url',
                  'license',
                  'flickr_url',
                  'height',
                  'width']
        img_df = pd.DataFrame.from_dict(img)\
                             .set_index('id')\
                             .drop(to_del, axis=1)
        img_df.rename_axis('image', inplace=True)

        output = anns_df.join(img_df, on='image')\
                        .drop(['area',
                               'segmentation',
                               'bbox'], axis=1)

        return output


class DatasetBuilder():
    def __init__(self, dir_, filename):
        self.dir_ = dir_
        self.filename = filename

    def _make_dataset(self, path='data/processed/images/*'):
        '''
        Makes the X dataset and corresponding filenames

        Input:

            path - the filepath of the preprocessed images

        Output:

            X - a NumPy array of the image arrays

            filename - a list of the corresponding file names
        '''
        image_dict = {'data': list(), 'filename': list()}

        for file in glob.glob(path):
            fname = 'images/' + file.split('/')[-1]
            img = io.imread(file)

            # gray = color.rgb2gray(img)
            # img_hsv = color.rgb2hsv(img)
            edges = sobel(img)

            image_dict['data'].append(edges.ravel())
            image_dict['filename'].append(fname)

        X_ = np.asarray(image_dict['data'])
        fnames_ = image_dict['filename']

        x_list = [i for i in zip(fnames_, X_)]
        dx = dict(sorted(x_list))
        sorted_dict = {'data': list(dx.values()), 'filenames': list(dx.keys())}

        X = np.asarray(sorted_dict['data'])
        fnames = sorted_dict['filenames']

        return X, fnames

    def _make_classes(self):
        '''
        Make the y target for the corresponding image set

        Input:

            just the output from the AnnotationsParser._json_to_df method

        Output:

            y - returns the target array
            filenames - returns the filenames for each target
            image_ids - returns the image ids for each target
        '''
        df_in = AnnotationsParser._json_to_df(self)
        out_dict = {'image': [], 'file_name': [], 'target': []}

        retail = []
        com = [2, 3, 5, 6]
        bad = [4]

        grouped = df_in.groupby('image')
        image_list = grouped.image.unique()
        for i in image_list:
            sub_group = grouped.get_group(int(i))
            image_categories = sub_group.category.unique()

            if np.any(np.isin(bad, image_categories)):
                target_ = 0
            elif np.any(np.isin(com, image_categories)):
                target_ = 1
            else:
                target_ = 2

            out_dict['image'].append(sub_group.iloc[0]['image'])
            out_dict['file_name'].append(sub_group.iloc[0]['file_name'])
            out_dict['target'].append(target_)

        y_ = np.asarray(out_dict['target'])
        filenames_ = out_dict['file_name']

        y_list = [i for i in zip(filenames_, y_)]
        dy = dict(sorted(y_list))
        sorted_dict = {'target': list(dy.values()),
                       'filenames': list(dy.keys())}

        y = np.asarray(sorted_dict['target'])
        filenames = sorted_dict['filenames']

        return y, filenames

    def _make_y(self):
        y_ = pd.read_csv('data/processed/y.csv').set_index('Unnamed: 0')
        y = y_['0'].to_numpy()

        return y

    def load_data(self):
        '''
        Makes our y classification targets
        '''
        X, fX = self._make_dataset()
        # y, fy = self._make_classes()
        y = self._make_y()

        return X, y


if __name__ == '__main__':
    src_dir = 'data/raw/annotations'
    fname = 'instances_default.json'

    lemons = DatasetBuilder(src_dir, fname)
    X, y = lemons.load_data()
