# This file will use the heaviest network to evaluate if the GPU can run the experiments.
from tensorflow import keras
from tensorflow.keras import layers

from utils.data_load_utils import load_mnist_data

batch_size = 32
num_filters = 256
kernel_size = 7
num_conv_layers = 10
num_pooling_layers_post_conv = 0
pool_kernel_size = 2
pool_stride = 2

num_of_classes = 10

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
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()

history = model.fit(x_train, y_train, epochs=5, batch_size=batch_size,
                    validation_data=(x_val, y_val))
