from keras.layers import BatchNormalization
from keras.optimizers import Adam
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models

from utils.data_load_utils import load_mnist_data, load_convex_data, load_rectangles, load_mnist_background_images, \
    load_mnist_background_random, load_mnist_rotation, load_mnist_rotation_background, load_rectangles_images

num_filters = 64
num_of_classes = 2
batch_size = 64
pool_kernel_size = 2
pool_stride = 2

image_input_shape = (28, 28, 1)
input = keras.Input(shape=image_input_shape)
x = input
x = layers.Conv2D(filters=32, kernel_size=7, activation='relu', padding='same')(x)
x = layers.Conv2D(filters=64, kernel_size=7, activation='relu', padding='same')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(num_of_classes, activation='softmax')(x)
fun_model = keras.Model(inputs=input, outputs=outputs)


seq_model = models.Sequential(
    [layers.Conv2D(filters=16, kernel_size=(9, 9), strides=(2, 2), activation='relu', padding='same', input_shape=(28, 28, 1)),
layers.Conv2D(filters=32, kernel_size=(7, 7), activation='relu', padding='same'),
layers.Conv2D(filters=64, kernel_size=(7, 7), activation='relu', padding='same'),
layers.Flatten(),
layers.Dense(64, activation='relu'),
layers.Dense(num_of_classes, activation='softmax')])

model = fun_model

model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_convex_data()
print('min pixel value: ' + str(x_train.min()))
print('max pixel value: ' + str(x_train.max()))


history = model.fit(x_train, y_train, epochs=10, batch_size=64,
                    validation_data=(x_val, y_val), shuffle=True)

score = model.evaluate(x_test, y_test, verbose=0)
predictions = model.predict(x_test, batch_size=64)
print('\n', 'Test accuracy:', score[1])