from keras.datasets import mnist


def load_mnist_data():
    input_width = 28
    input_height = 28
    input_channels = 1
    output_dim = 10

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    return (x_train, y_train), (x_test, y_test)
