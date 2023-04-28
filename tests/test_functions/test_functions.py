from evobpso.utils.data_load_utils import split_train_to_train_and_valid
from keras.datasets import mnist
import tensorflow as tf


def testdata_loader():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / x_train.max()
    x_test = x_test / x_test.max()

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)

    x_train = x_train[0:100, :, :]
    x_test = x_test[0:10, :, :]
    y_train = y_train[0:100, :]
    y_test = y_test[0:10, :]

    (x_train, y_train), (x_valid, y_valid) = split_train_to_train_and_valid(x_train=x_train, y_train=y_train)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)