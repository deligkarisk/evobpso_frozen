from pathlib import Path

from keras.datasets import mnist
import os
import numpy as np


def load_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    (x_train, y_train), (x_valid, y_valid) = split_train_to_train_and_valid(x_train=x_train, y_train=y_train)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_mnist_background_images():
    folder = os.path.join(get_project_root(), 'datasets', 'mnist_background_images')

    train = np.loadtxt(os.path.join(folder, 'mnist_background_images_train.amat'))
    test = np.loadtxt(os.path.join(folder, 'mnist_background_images_test.amat'))

    x_train = train[:, :-1]
    x_test = test[:, :-1]

    # Reshape images to 28x28
    x_train = np.reshape(x_train, (-1, 28, 28))
    x_test = np.reshape(x_test, (-1, 28, 28))

    y_train = train[:, -1]
    y_test = test[:, -1]

    (x_train, y_train), (x_valid, y_valid) = split_train_to_train_and_valid(x_train=x_train, y_train=y_train)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_mnist_background_random():
    folder = os.path.join(get_project_root(), 'datasets', 'mnist_background_random')

    train = np.loadtxt(os.path.join(folder, 'mnist_background_random_train.amat'))
    test = np.loadtxt(os.path.join(folder, 'mnist_background_random_test.amat'))

    x_train = train[:, :-1]
    x_test = test[:, :-1]

    # Reshape images to 28x28
    x_train = np.reshape(x_train, (-1, 28, 28))
    x_test = np.reshape(x_test, (-1, 28, 28))

    y_train = train[:, -1]
    y_test = test[:, -1]

    (x_train, y_train), (x_valid, y_valid) = split_train_to_train_and_valid(x_train=x_train, y_train=y_train)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_mnist_rotation():
    folder = os.path.join(get_project_root(), 'datasets', 'mnist_rotation_new')

    train = np.loadtxt(os.path.join(folder, 'mnist_all_rotation_normalized_float_train_valid.amat'))
    test = np.loadtxt(os.path.join(folder, 'mnist_all_rotation_normalized_float_test.amat'))

    x_train = train[:, :-1]
    x_test = test[:, :-1]

    # Reshape images to 28x28
    x_train = np.reshape(x_train, (-1, 28, 28))
    x_test = np.reshape(x_test, (-1, 28, 28))

    y_train = train[:, -1]
    y_test = test[:, -1]

    (x_train, y_train), (x_valid, y_valid) = split_train_to_train_and_valid(x_train=x_train, y_train=y_train)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_mnist_rotation_background():
    folder = os.path.join(get_project_root(), 'datasets', 'mnist_rotation_back_image_new')

    train = np.loadtxt(os.path.join(folder, 'mnist_all_background_images_rotation_normalized_train_valid.amat'))
    test = np.loadtxt(os.path.join(folder, 'mnist_all_background_images_rotation_normalized_test.amat'))

    x_train = train[:, :-1]
    x_test = test[:, :-1]

    # Reshape images to 28x28
    x_train = np.reshape(x_train, (-1, 28, 28))
    x_test = np.reshape(x_test, (-1, 28, 28))

    y_train = train[:, -1]
    y_test = test[:, -1]

    (x_train, y_train), (x_valid, y_valid) = split_train_to_train_and_valid(x_train=x_train, y_train=y_train)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_rectangles():
    folder = os.path.join(get_project_root(), 'datasets', 'rectangles')

    train = np.loadtxt(os.path.join(folder, 'rectangles_train.amat'))
    test = np.loadtxt(os.path.join(folder, 'rectangles_test.amat'))

    x_train = train[:, :-1]
    x_test = test[:, :-1]

    # Reshape images to 28x28
    x_train = np.reshape(x_train, (-1, 28, 28))
    x_test = np.reshape(x_test, (-1, 28, 28))

    y_train = train[:, -1]
    y_test = test[:, -1]

    (x_train, y_train), (x_valid, y_valid) = split_train_to_train_and_valid(x_train=x_train, y_train=y_train)

    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)


def load_rectangles_images():
    folder = os.path.join(get_project_root(), 'datasets', 'rectangles_images')

    train = np.loadtxt(os.path.join(folder, 'rectangles_im_train.amat'))
    test = np.loadtxt(os.path.join(folder, 'rectangles_im_test.amat'))

    x_train = train[:, :-1]
    x_test = test[:, :-1]

    # Reshape images to 28x28
    x_train = np.reshape(x_train, (-1, 28, 28))
    x_test = np.reshape(x_test, (-1, 28, 28))

    y_train = train[:, -1]
    y_test = test[:, -1]

    (x_train, y_train), (x_valid, y_valid) = split_train_to_train_and_valid(x_train=x_train, y_train=y_train)


    return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)



def load_convex_data():
    folder = os.path.join(get_project_root(), 'datasets', 'convex')

    train = np.loadtxt(os.path.join(folder, 'convex_train.amat'))
    test = np.loadtxt(os.path.join(folder, '50k', 'convex_test.amat'))

    x_train = train[:, :-1]
    x_test = test[:, :-1]

    # Reshape images to 28x28
    x_train = np.reshape(x_train, (-1, 28, 28))
    x_test = np.reshape(x_test, (-1, 28, 28))

    y_train = train[:, -1]
    y_test = test[:, -1]

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
