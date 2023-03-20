from evobpso.layer.Layer import ConvLayer, MaxPooling, AvgPooling
from evobpso.model_creator.ModelCreator import ModelCreator
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


class TensorflowModelCreator(ModelCreator):
    def create_model(self, architecture):

        input = keras.Input(shape=self.fixed_architecture_properties.input_shape)
        x = input

        for layer in architecture:
            if isinstance(layer, ConvLayer):
                x = layers.Conv2D(filters=layer.filters,
                                  kernel_size=layer.kernel_size,
                                  padding=self.fixed_architecture_properties.padding,
                                  strides=self.fixed_architecture_properties.conv_stride)(x)
                x = layers.BatchNormalization()(x)
                x = layers.Activation(activation=self.fixed_architecture_properties.activation_function)(x)
            elif isinstance(layer, MaxPooling):
                x = layers.MaxPooling2D(pool_size=self.fixed_architecture_properties.pool_layer_kernel_size,
                                        strides=self.fixed_architecture_properties.pool_layer_stride,
                                        padding=self.fixed_architecture_properties.padding)(x)
            elif isinstance(layer, AvgPooling):
                x = layers.AveragePooling2D(pool_size=self.fixed_architecture_properties.pool_layer_kernel_size,
                                            strides=self.fixed_architecture_properties.pool_layer_stride,
                                            padding=self.fixed_architecture_properties.padding)(x)

        x = layers.Flatten()(x)
        outputs = layers.Dense(self.fixed_architecture_properties.dense_layer_units, activation="softmax")(x)
        model = keras.Model(inputs=input, outputs=outputs)
        return model








