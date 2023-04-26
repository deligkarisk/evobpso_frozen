# This file will use the heaviest network to evaluate if the GPU can run the experiments.
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf

from utils.data_load_utils import load_mnist_data

batch_size = 64
num_filters = 256
kernel_size = 7
num_conv_layers = 20
num_pooling_layers_post_conv = 0
pool_kernel_size = 2
pool_stride = 2

num_of_classes = 10





physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
tf.config.set_visible_devices(physical_devices[0], 'GPU')

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

for i in range(0,20):
    print('iteration: ' + str(i))
    image_input_shape = (28, 28, 1)
    input = keras.Input(shape=image_input_shape)
    x = layers.Rescaling(1. / 255)(input)

    for i in range(0, num_conv_layers):
        x = layers.Conv2D(filters=num_filters,
                          kernel_size=7,
                          activation='relu',
                          padding='same')(x)
        for k in range(0, num_pooling_layers_post_conv):
            x = layers.MaxPooling2D(pool_size=pool_kernel_size,
                                    strides=pool_stride,
                                    padding='same')(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(num_of_classes, activation='softmax')(x)
    model = keras.Model(inputs=input, outputs=outputs)
    # model.summary()
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()

    history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size,
                        validation_data=(x_val, y_val))
