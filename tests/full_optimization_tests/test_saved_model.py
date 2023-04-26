# This file will use the heaviest network to evaluate if the GPU can run the experiments.
import tensorflow as tf

from utils.data_load_utils import load_mnist_data
from utils.utils import set_seed

batch_size = 64
num_filters = 256
kernel_size = 7
num_conv_layers = 10
num_pooling_layers_post_conv = 0
pool_kernel_size = 2
pool_stride = 2

num_of_classes = 10

image_input_shape = (28, 28, 1)

set_seed(15)
model = tf.keras.models.load_model('//test_tmp_results/iter_1/particle_0')

model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='rmsprop', metrics=['accuracy'])

(x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()

history = model.fit(x_train, y_train, epochs=1, batch_size=batch_size,
                    validation_data=(x_val, y_val))
