import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model


def make_generators(direc, image_size, batch_size, split_size=0.2):
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
            shear_range=0.2,
            zoom_range=0.2,
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
            subset='training'
            )

    validation_generator = test_datagen.flow_from_directory(
            directory=direc,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation'
            )

    return train_generator, validation_generator


if __name__ == '__main__':
    direc = 'data/raw/classified'

    image_size = (299, 299)
    batch_size = 32

    inputs = tf.keras.layers.Input(shape=image_size + (3,), dtype = tf.uint8)
    x = tf.cast(i, tf.float32)
    x = tf.keras.applications.mobilenet.preprocess_input(x)
    core = tf.keras.applications.MobileNet()
    x = core(x)
    model = tf.keras.Model(inputs=[i], outputs=[x])

    image = tf.image.decode_png(tf.io.read_file('file.png'))
    result = model(image)

    X_train, X_test = make_generators(direc, image_size, batch_size)

    tf.keras.applications.xception.preprocess_input(
            X_train,
            data_format=None
            )

    model = tf.keras.applications.Xception(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3,
            classifier_activation='softmax'
            )

    keras.utils.plot_model(model, show_shapes=True)

    callbacks = [
            keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),
            ]

    model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=["accuracy"],
            )

    class_weights = {
            0: 1,
            1: 4.4,
            2: 1.03
            }

    model.fit(
            X_train,
            validation_data=X_test,
            callbacks=callbacks,
            steps_per_epoch=2000,
            epochs=50,
            validation_steps=800
            )

    model_file = 'models/xception.hd5'

    model.save(model_file)
    del model

    model = load_model(model_file)