import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImagePreprocessing:
    def __init__(self):
        pass
    def make_generators(self, direc, image_size, batch_size=32, split_size=0.2):
        '''
        Makes the train & validataion datasets.

        Parameters
        ----------
        direc: string
                Location of the images, in folders according to their classifications
        image_size: tuple
                Size of the images to send into the CNN
        batch_size: int
                The batch size the generator will build
        split_size: float
                The size of the validation dataset. Default = 0.2 (20%)
        
        Output
        ------
        train_generator: generator
                Generator containing the training dataset    
        validation_generator: generator
                Generator containing the validation dataset

        '''
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.4,
                channel_shift_range=0.4,
                zoom_range=0.4,
                horizontal_flip=True,
                validation_split=split_size
                )
        
        test_datagen = ImageDataGenerator(
                rescale=1./255,
                validation_split=split_size
                )

        train_generator = train_datagen.flow_from_directory(
                directory=direc,
                target_size=image_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='training',
                shuffle=True
                )

        validation_generator = test_datagen.flow_from_directory(
                directory=direc,
                target_size=image_size,
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation',
                shuffle=True
                )

        return train_generator, validation_generator

    def preprocess_single(self, image):

        pass