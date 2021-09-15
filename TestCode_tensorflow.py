# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 23:16:11 2021

@author: Moha-Esla
"""


##############################
# check the tensorflow and device
###########
import tensorflow as tf

if tf.test.is_gpu_available():
    gpus = tf.config.list_physical_devices('GPU')
    print("========> tensorflow version:", tf.__version__ ," <========")
    print("========> GPU found <========")
    print("========>  Number of Available GPUs: ", len(gpus) , ' <========') 
    for gpu in gpus:
        print("========> Name:", gpu.name, "  Type:", gpu.device_type, ' <========')
else:
    print("========> tensorflow version:", tf.__version__ ," <========")
    print(("========> GPU is not available. CPU will be used! If you dont want, stop it manually <========"))


##############################
# training code: https://keras.io/examples/vision/mnist_convnet/
##################

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
num_classes = 10
input_shape = (28, 28, 1)

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Scale images to the [0, 1] range
x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255
# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)
print("x_train shape:", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")


# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()

batch_size = 128
epochs = 15

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

