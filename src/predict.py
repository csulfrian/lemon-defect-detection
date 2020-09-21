import numpy as np
import os
# import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report

from xception import make_generators


def predict_one(model, path):
    image_size = (299, 299)

    img = tf.keras.preprocessing.image.load_img(path, target_size=image_size)
    input_arr = keras.preprocessing.image.img_to_array(img)
    input_arr = tf.expand_dims(input_arr, 0)
    img=tf.keras.applications.xception.preprocess_input(input_arr)
    
    print('Predicting!\n')
    pred = model.predict(img, verbose=0)

    print(f'Probabilities: {np.around(pred, 2)}')
    category = np.argmax(pred, axis=-1)
    score = tf.nn.softmax(pred[0])
    best = np.max(score)

    print(f'Image {img_name} most likely belongs to class {category[0]}, with {(best * 100):0.0f}% confidence.\n')

def predict_batch(model, batch):
    print('Predicting!')
    y_pred = model.predict(batch, verbose=1)
    probs = np.around(y_pred, 2)

    category = np.argmax(y_pred, axis=-1)

    for i in zip(batch.class_indices, category, probs):
        print(i)

    report = classification_report(batch.classes, category)
    print(report)


if __name__ == '__main__':
    tf.get_logger().setLevel('ERROR')
    
    print('Loading model...')
    model = keras.models.load_model('models/xception_transfer_seeded', compile=True)

    file_path = 'data/created/'
    img_name = 'DSCN3580.JPG'

    # predict_one(model, file_path+img_name)

    direc = 'data/raw/classified'
    image_size = (299, 299)
    batch_size = 32

    _, X_test = make_generators(direc, image_size, batch_size)
    # print(X_test.classes)

    predict_batch(model, X_test)