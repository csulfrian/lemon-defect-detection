import os
import numpy as np
import matplotlib.pyplot as plt
import re
import random
# import cv2

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from xception import make_generators

tf.get_logger().setLevel('ERROR')

# train_dir = os.path.join(PATH, 'train')
# validation_dir = os.path.join(PATH, 'validation')

# train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
# train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
# validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
# validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

# cats_tr = os.listdir(train_cats_dir)
# dogs_tr = os.listdir(train_dogs_dir)

# cats_val = os.listdir(validation_cats_dir)
# dogs_val = os.listdir(validation_dogs_dir)

# cats_tr = [os.path.join(train_cats_dir, x) for x in cats_tr]
# dogs_tr = [os.path.join(train_dogs_dir, x) for x in dogs_tr]
# cats_val = [os.path.join(validation_cats_dir, x) for x in cats_val]
# dogs_val = [os.path.join(validation_dogs_dir, x) for x in dogs_val]

# total_train = cats_tr + dogs_tr
# total_val = cats_val + dogs_val

# random.shuffle(total_train)
# X_train = np.zeros((len(total_train), 224, 224, 3)).astype('float')
# y_train = []
# for i, img_path in enumerate(total_train):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (224, 224))
#     X_train[i] = img
#     if len(re.findall('dog', img_path)) == 3:
#         y_train.append(0)
#     else: 
#         y_train.append(1)
# y_train = np.array(y_train)

# random.shuffle(total_val)
# X_test = np.zeros((len(total_val), 224, 224, 3)).astype('float')
# y_test = []
# for i, img_path in enumerate(total_val):
#     img = cv2.imread(img_path)
#     img = cv2.resize(img, (224, 224))
#     X_test[i] = img
#     if len(re.findall('dog', img_path)) == 3:
#         y_test.append(0)
#     else: 
#         y_test.append(1)
# y_test = np.array(y_test)

direc = 'data/raw/classified'
image_size = (224, 224)
batch_size = 32

X_train, X_test = make_generators(direc, image_size, batch_size)

epochs = 30
IMG_HEIGHT = image_size[1]
IMG_WIDTH = image_size[0]

def create_model(base_model):
    base_model.trainable = True
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    prediction_layer = tf.keras.layers.Dense(3,
            activation='softmax',
            kernel_regularizer=tf.keras.regularizers.l2(0.0001))(global_average_layer)
    model = tf.keras.models.Model(inputs=base_model.input,
            outputs=prediction_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["categorical_accuracy", "Recall", "AUC"])
    return model

def fit_model(model):
    class_weights = {
            0: 1.03,
            1: 4.4,
            2: 1
            }
    history = model.fit(
                        X_train,
                        batch_size=batch_size,
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=epochs,
                        class_weight=class_weights,
                        validation_data=X_test,
                        validation_steps=len(X_test) // batch_size
                        )
    return history

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss=history.history['loss']
    val_loss=history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

IMG_SHAPE = (224, 224, 3)
base_model1 = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
base_model2 = tf.keras.applications.InceptionV3(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")
base_model3 = tf.keras.applications.Xception(input_shape=IMG_SHAPE, include_top=False, weights="imagenet")

model1 = create_model(base_model1)
model2 = create_model(base_model2)
model3 = create_model(base_model3)

history1 = fit_model(model1)
model1.save('models/model1.h5')

history2 = fit_model(model2)
model2.save('models/model2.h5')

history3 = fit_model(model3)
model3.save('models/model3.h5')

# plot_history(history1)
# plot_history(history2)
# plot_history(history3)

def load_all_models():
    all_models = []
    model_names = ['model1.h5', 'model2.h5', 'model3.h5']
    for model_name in model_names:
        filename = os.path.join('models', model_name)
        model = tf.keras.models.load_model(filename)
        all_models.append(model)
        print('loaded:', filename)
    return all_models

def ensemble_model(models):
    for i, model in enumerate(models):
        for layer in model.layers:
            layer.trainable = False
    ensemble_visible = [model.input for model in models]
    ensemble_outputs = [model.output for model in models]
    merge = tf.keras.layers.concatenate(ensemble_outputs)
    merge = tf.keras.layers.Dense(10, activation='relu')(merge)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(merge)
    model = tf.keras.models.Model(inputs=ensemble_visible, outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
    return model

models = load_all_models()
model = ensemble_model(models)

X = [X_train for _ in range(len(model.input))]
X_1 = [X_test for _ in range(len(model.input))]

epochs = 20
history = model.fit(X_train,
                    batch_size=batch_size,
                    steps_per_epoch=len(X_train) // batch_size,
                    epochs=epochs,
                    validation_data=X_test,
                    validation_steps=len(X_test) // batch_size
                    )

model.save('models/model.h5')

plot_history(history)

print('MobileNetV2 acc:', history1.history['val_recall'][-1])
print('InceptionV3 acc:', history2.history['val_recall'][-1])
print('Xception acc:', history3.history['val_recall'][-1])
print('Ensemble acc:', history.history['val_recall'][-1])