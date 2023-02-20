from unittest import TestCase
from unittest.mock import Mock

from keras.engine.input_layer import InputLayer
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.python.framework.tensor_shape import Dimension

from layer.Layer import ConvLayer, MaxPooling
from model_creator.TensorflowModelCreator import TensorflowModelCreator
from params.FixedArchitectureParams import FixedArchitectureParams


class TestTensorflowModelCreator(TestCase):

    def test_sample_architecture_1(self):
        mock_architecture_params = Mock()
        mock_architecture_params.input_shape = (180, 180, 3)
        mock_architecture_params.conv_stride = 2
        mock_architecture_params.activation_function = 'relu'
        mock_architecture_params.pool_layer_kernel_size = 2
        mock_architecture_params.pool_layer_stride = 4
        mock_architecture_params.padding = 'same'
        mock_architecture_params.dense_layer_units = 10
        model_creator = TensorflowModelCreator(mock_architecture_params)

        architecture = [ConvLayer(32, 3), MaxPooling(), ConvLayer(64, 6), MaxPooling(), ConvLayer(128, 9)]
        model = model_creator.create_model(architecture=architecture)
        model.summary()

        assert model.input_shape == (None,) + mock_architecture_params.input_shape
        assert model.output_shape == (None,) + (mock_architecture_params.dense_layer_units,)

        assert isinstance(model.layers[0], InputLayer)
        assert isinstance(model.layers[1], Rescaling)
        assert isinstance(model.layers[2], Conv2D)
        assert isinstance(model.layers[3], MaxPooling2D)
        assert isinstance(model.layers[4], Conv2D)
        assert isinstance(model.layers[5], MaxPooling2D)
        assert isinstance(model.layers[6], Conv2D)
        assert isinstance(model.layers[7], Flatten)
        assert isinstance(model.layers[8], Dense)

        assert model.layers[2].filters == 32
        assert model.layers[2].kernel_size == (3, 3)
        assert model.layers[2].padding == 'same'
        assert model.layers[3].pool_size == (2, 2)
        assert model.layers[3].strides == (4, 4)
        assert model.layers[3].padding == 'same'
        assert model.layers[4].filters == 64
        assert model.layers[4].kernel_size == (6, 6)
        assert model.layers[4].padding == 'same'
        assert model.layers[5].pool_size == (2, 2)
        assert model.layers[5].strides == (4, 4)
        assert model.layers[5].padding == 'same'
        assert model.layers[6].filters == 128
        assert model.layers[6].kernel_size == (9, 9)
        assert model.layers[6].padding == 'same'
        assert model.layers[8].output_shape == (None, mock_architecture_params.dense_layer_units)

