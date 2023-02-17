from layer.Layer import ConvLayer, MaxPooling, AvgPooling
from model_creator.ModelCreator import ModelCreator
from tensorflow import keras
from tensorflow.keras import layers


class TensorflowModelCreator(ModelCreator):
    def create_model(self, architecture):

        input = keras.Input(shape=(180, 180, 3))
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
                x = layers.AvgPooling2D(pool_size=self.fixed_architecture_params.pool_layer_kernel_size,
                                        strides=self.fixed_architecture_params.pool_layer_stride,
                                        padding=self.fixed_architecture_params.padding)(x)


            outputs = layers.Dense(10, activation="softmax")(x)
            model = keras.Model(inputs=inputs, outputs=outputs)




        raise NotImplementedError



