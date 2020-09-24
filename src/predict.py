import numpy as np
import os
import glob
from skimage import io
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from xception import make_generators


def predict_one(model, path):
    '''
    Predict one image's classification. Prints the probabilities
    and the class prediction.

    Parameters
    ----------
    model: Keras model
        Feed in the loaded model file you want to predict on.
    path: string
        The path of the single image to predict on.
        No pre-processing necessary

    Outputs
    -------
    category: int
        The predicted category of the image
    '''
    image_size = (299, 299)

    img = tf.keras.preprocessing.image.load_img(
        path,
        target_size=image_size
    )
    input_arr = keras.preprocessing.image.img_to_array(img)
    input_arr = tf.expand_dims(input_arr, 0)
    img = tf.keras.applications.xception.preprocess_input(input_arr)

    print('Predicting!\n')
    pred = model.predict(img, verbose=0)

    print(f'Probabilities: {np.around(pred, 2)}')
    category = np.argmax(pred, axis=-1)
    score = tf.nn.softmax(pred[0])
    best = np.max(score)

    print(f'Image {path} most likely belongs to class \
          {category[0]}, with {(best * 100):0.0f}% confidence.\n')

    return category


def predict_batch(model, directory='data/raw/classified', print_metrics=False):
    '''
    Predict one image's classification. Prints the probabilities
    and the class prediction

    Parameters
    ----------
    model: Keras model
        Feed in the loaded model file you want to predict on
    directory: string
        The path of the directory containing the images to predict on

    Outputs
    -------
    category: array
        The predicted categories of the images
    '''
    directory = directory
    image_size = (299, 299)
    batch_size = 32

    X_test = make_generators(
        directory,
        image_size,
        batch_size,
    )

    print('Predicting!')
    y_pred = model.predict(X_test, verbose=1)
    probabilities = np.around(y_pred, 2)

    categories = np.argmax(y_pred, axis=-1)
    if print_metrics:
        report = classification_report(X_test.classes, categories)
        print(report)

        conf = confusion_matrix(X_test.classes, categories)
        print(conf)

    for i in zip(X_test.classes, categories):
        print(i)
    return categories


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')

    print('Loading model...')
    model = keras.models.load_model(
        'models/xception_post_transfer.hd5',
        compile=True
    )

    for file in glob.glob('data/created/*'):
        img = io.imread(file)
        predict_one(model, img)
        # print(file)