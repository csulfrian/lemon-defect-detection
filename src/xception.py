import numpy as np
import os
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model

tf.get_logger().setLevel('ERROR')

def make_generators(direc, image_size, batch_size=32, split_size=0.2):
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


def make_model(transfer=False, parallel=False):
    strategy = parallelize()

    if transfer:
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
            x = keras.layers.GlobalAveragePooling2D()(x)
            outputs = keras.layers.Dense(
                    units=3,
                    activation='softmax',
                    kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
            model = keras.Model(inputs, outputs)

            model.compile(
                optimizer=keras.optimizers.Adam(
                learning_rate=0.001,
                epsilon=0.1),
                loss="categorical_crossentropy",
                metrics=["categorical_accuracy", "Recall", "AUC"]
            )
        return model

    elif not transfer:
        with strategy.scope():
            model = tf.keras.applications.Xception(
                include_top=True,
                weights=None,
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=3
            )

            model.trainable = True

            model.compile(
                optimizer=keras.optimizers.Adam(
                learning_rate=0.001,
                epsilon=0.1),
                loss="categorical_crossentropy",
                metrics=["categorical_accuracy", "Recall", "AUC"]
            )
        return model


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
      # Restrict TensorFlow to only allocate 12GB of memory on the first GPU
      try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=15288)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

    direc = 'data/raw/classified'
    image_size = (299, 299)
    batch_size = 32

    X_train, X_test = make_generators(direc, image_size, batch_size)

    # model = make_model(transfer=True)
    strategy = parallelize()
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
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(units=3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
        model = keras.Model(inputs, outputs)

        # model.summary()

        model.compile(
            optimizer=keras.optimizers.Adam(
            learning_rate=0.001,
            epsilon=0.1),
            loss='CategoricalCrossentropy',
            metrics=["categorical_accuracy", "Recall"]
        )

    checkpoint_filename = 'models/save_at_{epoch}.h5'
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filename,
            monitor='val_loss',
            mode='max',
            save_best_only=True
            ),
        keras.callbacks.TensorBoard(
            log_dir='models',
            histogram_freq=1,
            write_graph=True,
            embeddings_freq=0)
        #     ),
        # keras.callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     min_delta=0.003,
        #     patience=3,
        #     verbose=1,
        #     mode='max',
        #     baseline=None,
        #     restore_best_weights=True
        #     )
        ]

    class_weights = {
                0: 1.03,
                1: 4.4,
                2: 1
                }

    model.fit(
        X_train,
        validation_data=X_test,
        callbacks=callbacks,
        steps_per_epoch=None,
        class_weight=class_weights,
        epochs=35,
        verbose=1
	)

    model_file = 'models/xception.hd5'

    model.save(filepath=model_file, include_optimizer=True)
    # del model

    # model = load_model(model_file)


    # Unfreeze the base model
    base_model.trainable = True

    # model.summary()


    # It's important to recompile your model after you make any changes
    # to the `trainable` attribute of any inner layer, so that your changes
    # are taken into account
    with strategy.scope():
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss='CategoricalCrossentropy',
            metrics=["categorical_accuracy", "Recall", "AUC"]
        )

    # Train end-to-end. Be careful to stop before you overfit!
    model.fit(
        X_train,
        validation_data=X_test,
        callbacks=callbacks,
        steps_per_epoch=None,
        class_weight=class_weights,
        epochs=5,
        verbose=1
	)

    model.save(filepath='models/xception_post_transfer.hd5', include_optimizer=True)
