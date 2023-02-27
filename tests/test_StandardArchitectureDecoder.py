from unittest import TestCase

from architecture_decoder.StandardArchitectureDecoder import StandardArchitectureDecoder
from layer.Layer import AvgPooling, MaxPooling, ConvLayer


class TestStandardArchitectureDecoder(TestCase):
    def test_decode(self):
        position = []
        position.append(0b11001110000010)
        filters_first_dim = 131  # binary is 130, then adding one as the minimum filter is one, not zero.
        kernel_size_first_dim = 5  # binary is three, adding two as the minimum kernel size is two.
        pooling_layers_first_dim = 2
        pooling_layer_type_first_dim = AvgPooling()  # a value of one is average pooling
        position.append(0b01111100010010)
        filters_second_dim = 19  # binary is 18, then adding one as the minimum filter is one, not zero.
        kernel_size_second_dim = 9  # binary is three, adding two as the minimum kernel size is two.
        pooling_layers_second_dim = 3
        pooling_layer_type_second_dim = MaxPooling()  # a value of zero is max pooling
        position.append(0b10000000000011)
        filters_third_dim = 4  # binary is three, then adding one as the minimum filter is one, not zero.
        kernel_size_third_dim = 2  # binary is zero, adding two as the minimum kernel size is two.
        pooling_layers_third_dim = 0  # no pooling layer after the third convolutional layer.

        decoder = StandardArchitectureDecoder()
        returned_architecture = decoder.decode(position)

        assert len(returned_architecture) == 8  # Conv, Pooling, Pooling, Conv, Pooling, Pooling, Pooling, Conv
        assert isinstance(returned_architecture[0], ConvLayer)
        assert returned_architecture[0].filters == filters_first_dim
        assert returned_architecture[0].kernel_size == kernel_size_first_dim
        assert isinstance(returned_architecture[1], AvgPooling)
        assert isinstance(returned_architecture[2], AvgPooling)
        assert isinstance(returned_architecture[3], ConvLayer)
        assert returned_architecture[3].filters == filters_second_dim
        assert returned_architecture[3].kernel_size == kernel_size_second_dim
        assert isinstance(returned_architecture[4], MaxPooling)
        assert isinstance(returned_architecture[5], MaxPooling)
        assert isinstance(returned_architecture[6], MaxPooling)
        assert isinstance(returned_architecture[7], ConvLayer)
        assert returned_architecture[7].filters == filters_third_dim
        assert returned_architecture[7].kernel_size == kernel_size_third_dim
