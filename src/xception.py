import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model
from frozen_graph_gen import save_frozen_model


def make_generators(direc, image_size, batch_size=32, split_size=0.2):
    """
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

    Outputs
    -------
    train_generator: generator
            Generator containing the training dataset
    validation_generator: generator
            Generator containing the validation dataset

    """
    train_datagen = ImageDataGenerator(
        data_format="channels_last",
        rescale=1.0 / 255,
        shear_range=0.4,
        channel_shift_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True,
        validation_split=split_size,
    )

    test_datagen = ImageDataGenerator(
        data_format="channels_last",
        rescale=1.0 / 255,
        validation_split=split_size
    )

    train_generator = train_datagen.flow_from_directory(
        directory=direc,
        seed=42,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    validation_generator = test_datagen.flow_from_directory(
        directory=direc,
        seed=42,
        target_size=image_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
    )

    return train_generator, validation_generator


def parallelize(hardware="GPU"):
    """
    Initializes the ability to use a mirroring strategy on a GPU or TPU

    Parameters
    ----------
    hardware: string
            The kind of hardware you would like to use. Default = 'GPU'

    Output
    ------
    strategy: strategy object
            The mirroring strategy for the .compile method to build with.
    """
    if hardware == "TPU":
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu="grpc://" + os.environ["COLAB_TPU_ADDR"]
        )
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        print("All devices: ", tf.config.list_logical_devices("TPU"))

        strategy = tf.distribute.TPUStrategy(resolver)
        return strategy

    if hardware == "GPU":
        strategy = tf.distribute.MirroredStrategy()
        return strategy


def make_model(transfer=False, train_able=True):
    """
    Builds a compiled model, with or without transfer learning

    Parameters
    ----------
    transfer: boolean
        True enables transfer learning. Default = False

    Output
    ------
    model: Keras model
        The compiled model
    """
    if transfer:
        base_model = tf.keras.applications.Xception(
            include_top=False,
            weights="imagenet",
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3,
        )

        base_model.trainable = train_able

        inputs = keras.Input(shape=(3, 299, 299))
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)

        outputs = keras.layers.Dense(
            units=3,
            activation="softmax",
            kernel_regularizer=tf.keras.regularizers.l2(0.0001),
        )(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.00001, epsilon=0.1),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy", "Recall", "AUC"],
        )
        return model

    elif not transfer:
        model = tf.keras.applications.Xception(
            include_top=True,
            weights=None,
            input_tensor=None,
            input_shape=None,
            pooling=None,
            classes=3,
        )

        model.trainable = train_able

        model.compile(
            optimizer=keras.optimizers.Adam(
                learning_rate=0.00001, epsilon=0.1),
            loss="categorical_crossentropy",
            metrics=["categorical_accuracy", "Recall", "AUC"],
        )
        return model


def main():
    direc = "data/raw/classified"
    image_size = (299, 299)
    batch_size = 20

    class_weights = {0: 1.03,
                     1: 4.4,
                     2: 1
                     }

    X_train, X_test = make_generators(direc, image_size, batch_size)

    checkpoint_filename = "models/save_at_{epoch}.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filename,
            monitor="val_loss",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.TensorBoard(
            log_dir="logs",
            histogram_freq=1,
            write_images=True,
            write_graph=True,
            embeddings_freq=1,
        ),
    ]

    # model = make_model(transfer=True, train_able=True)
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=3,
    )

    base_model.trainable = False

    inputs = keras.Input(shape=(3, 299, 299))
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(
        units=3,
        activation="softmax",
        kernel_regularizer=tf.keras.regularizers.l2(0.0001),
    )(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001, epsilon=0.1),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "Recall", "AUC"],
    )

    model.trainable = False

    model.fit(
        X_train,
        validation_data=X_test,
        callbacks=callbacks,
        steps_per_epoch=None,
        class_weight=class_weights,
        epochs=1,
        verbose=1
    )

    model.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="categorical_crossentropy",
        metrics=["categorical_accuracy", "Recall", "AUC"],
    )

    model.fit(
        X_train,
        validation_data=X_test,
        callbacks=callbacks,
        steps_per_epoch=None,
        class_weight=class_weights,
        epochs=1,
        verbose=1
    )

    save_frozen_model(model, "xception_channels_first")

    model.save(filepath='models/xception_transfer', include_optimizer=True)
    del model


if __name__ == "__main__":
    # Keras housekeeping - must happen before anything else!
    keras.backend.set_image_data_format("channels_first")
    print("Channel mode:", keras.backend.image_data_format())
    tf.get_logger().setLevel("ERROR")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)

    main()
