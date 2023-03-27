from pathlib import Path

import tensorflow as tf
from keras.datasets import mnist


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / x_train.max()
    x_test = x_test / x_test.max()

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    (x_train, y_train), (x_valid, y_valid) = split_train_to_train_and_valid(x_train=x_train, y_train=y_train)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def split_train_to_train_and_valid(x_train, y_train):
    valid_data = int(0.2 * len(x_train))
    x_valid = x_train[0:valid_data, ]
    y_valid = y_train[0:valid_data, ]
    x_train = x_train[valid_data:, ]
    y_train = y_train[valid_data:, ]

    return (x_train, y_train), (x_valid, y_valid)

def get_project_root():
    thisfile = Path(__file__)
    root_dir = thisfile.parent.parent
    return root_dir
