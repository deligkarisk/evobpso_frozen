from layer.Layer import ConvLayer, MaxPooling, AvgPooling
from model_creator.ModelCreator import ModelCreator
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class TensorflowModelCreator(ModelCreator):
    def create_model(self, architecture):

        input = keras.Input(shape=self.fixed_architecture_params.input_shape)
        x = layers.Rescaling(1. / 255)(input)

        for layer in architecture:
            if isinstance(layer, ConvLayer):
                x = layers.Conv2D(filters=layer.filters,
                                  kernel_size=layer.kernel_size,
                                  activation=self.fixed_architecture_params.activation_function,
                                  padding=self.fixed_architecture_params.padding)(x)
            elif isinstance(layer, MaxPooling):
                x = layers.MaxPooling2D(pool_size=self.fixed_architecture_params.pool_layer_kernel_size,
                                        strides=self.fixed_architecture_params.pool_layer_stride,
                                        padding=self.fixed_architecture_params.padding)(x)
            elif isinstance(layer, AvgPooling):
                x = layers.AveragePooling2D(pool_size=self.fixed_architecture_params.pool_layer_kernel_size,
                                        strides=self.fixed_architecture_params.pool_layer_stride,
                                        padding=self.fixed_architecture_params.padding)(x)

        x = layers.Flatten()(x)
        outputs = layers.Dense(self.fixed_architecture_params.dense_layer_units, activation="softmax")(x)
        model = keras.Model(inputs=input, outputs=outputs)
        return model








