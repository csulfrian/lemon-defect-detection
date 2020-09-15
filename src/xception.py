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

def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
                plt.ylim([0.8,1])
        else:
                plt.ylim([0,1])

        plt.legend()

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
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*6)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    direc = 'data/raw/classified'
    image_size = (299, 299)
    batch_size = 32

    # strategy = parallelize()

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
            monitor='val_loss',
            min_delta=0.002,
            patience=1,
            verbose=2,
            mode='auto',
            baseline=None,
            restore_best_weights=True
            )

        ]

    # with strategy.scope():
    model = tf.keras.applications.Xception(
    include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=3
        )

    model.compile(
    optimizer=keras.optimizers.Adam(
    learning_rate=0.001,
        epsilon=0.1),
        loss="categorical_crossentropy",
        metrics=["CategoricalAccuracy", "Recall"]
    )

        # experimental_steps_per_execution = 50,
        # keras.utils.plot_model(model, show_shapes=True)

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
        epochs=20,
        class_weight=class_weights,
        verbose=2
	)

    model_file = 'models/xception.hd5'

    model.save(model_file)
    del model

    model = load_model(model_file)
