import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model

from xception import make_generators

if __name__ == '__main__':
    print('Loading model...')
    model = load_model('models/xception.hd5')
    
    print('Making dataset...')
    direc = 'data/raw/classified'
    image_size = (299, 299)
    batch_size = 32

    X_train, X_test = make_generators(direc, image_size, batch_size)
    print(X_test[0])

    # print('Predicting!')
    # preds = np.argmax(model.predict(X_test, verbose=1), axis=-1)

    # fig = plt.figure(figsize=(20, 20))

    # for i, img in enumerate(X_test):
    #     fig.add_subplot(4,5, i+1)
    #     # plt.imshow(img)
    
    # plt.savefig('images/sample_predict.png')

    # print(np.unique(preds, return_counts=True))
    # for i in zip(X_test[1], preds):
    #     print(i)
