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

tf.get_logger().setLevel('ERROR')

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
            channel_shift_range=0.2,
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

def parallelize(hardware='GPU'):
    '''
    Initializes the ability to use a mirroring strategy on a GPU or TPU

    Parameters
    ----------
    hardware: string
            The kind of hardware you would like to use. Default = 'GPU'

    Output
    ------
    strategy: strategy object
            The mirroring strategy for the .compile method to build with.
    '''
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    if hardware == 'TPU':
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices('TPU'))

        strategy = tf.distribute.TPUStrategy(resolver)
        return strategy

    if hardware == 'GPU':
        strategy = tf.distribute.MirroredStrategy()
        tf.config.experimental.list_physical_devices('GPU')
        return strategy


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12288)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    direc = 'data/raw/classified'
    image_size = (299, 299)
    batch_size = 32

    strategy = parallelize()

    X_train, X_test = make_generators(direc, image_size, batch_size)

    checkpoint_filename = 'models/save_at_{epoch}.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filename,
            monitor='categorical_crossentropy',
            mode='auto',
            save_best_only=True
            ),
        keras.callbacks.TensorBoard(
            log_dir='models',
            histogram_freq=0,
            write_graph=True,
            embeddings_freq=0
            ),
        keras.callbacks.EarlyStopping(
            monitor='CategoricalAccuracy',
            min_delta=0.0005,
            patience=4,
            verbose=1,
            mode='max',
            baseline=None,
            restore_best_weights=True
            )
        ]

    with strategy.scope():
        base_model = tf.keras.applications.Xception(
            include_top=False,
            weights='imagenet',
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3
        )
        
        base_model.trainable = False

        inputs = keras.Input(shape=(299, 299, 3))
        x = base_model(inputs, training=False)
        x = keras.layers.Dense(units=3, activation='softmax')(x)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(
            learning_rate=0.001,
            epsilon=0.1),
            loss="categorical_crossentropy",
            metrics=["CategoricalAccuracy", "Recall"]
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
        steps_per_epoch=None,
        epochs=50,
        class_weight=class_weights,
        verbose=1
	)

    model_file = 'models/xception.hd5'

    model.save(filepath=model_file, include_optimizer=True)
    # del model

    # model = load_model(model_file)