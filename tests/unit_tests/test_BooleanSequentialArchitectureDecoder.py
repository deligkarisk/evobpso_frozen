from unittest import TestCase

from evobpso.architecture_decoder.BooleanSequentialArchitectureDecoder import BooleanSequentialArchitectureDecoder
from evobpso.encoding.BooleanEncoding import BooleanEncoding
from evobpso.layer.Layer import AvgPooling, MaxPooling, ConvLayer, FlattenLayer, DenseLayer


class TestBooleanSequentialArchitectureDecoder(TestCase):
    def test_decode_with_boolean_sample_encoding(self):

        encoding = BooleanEncoding(filter_bits=8, kernel_size_bits=3)
        position = []
        position.append(0b1101110000010)
        filters_first_dim = 131  # binary is 130, then adding one as the minimum filter is one, not zero.
        kernel_size_first_dim = 5  # binary is three, adding two as the minimum kernel size is two.
        pooling_layers_first_dim = 1
        pooling_layer_type_first_dim = AvgPooling()  # a value of one is average pooling
        position.append(0b0111100010010)
        filters_second_dim = 19  # binary is 18, then adding one as the minimum filter is one, not zero.
        kernel_size_second_dim = 9  # binary is three, adding two as the minimum kernel size is two.
        pooling_layers_second_dim = 1
        pooling_layer_type_second_dim = MaxPooling()  # a value of zero is max pooling
        position.append(0b1000000000011)
        filters_third_dim = 4  # binary is three, then adding one as the minimum filter is one, not zero.
        kernel_size_third_dim = 2  # binary is zero, adding two as the minimum kernel size is two.
        pooling_layers_third_dim = 0  # no pooling layer after the third convolutional layer.

        decoder = BooleanSequentialArchitectureDecoder(encoding=encoding)
        returned_architecture = decoder.decode(position)

        assert len(returned_architecture) == 7  # Conv, Pooling, Conv, Pooling, Conv, Flatten, Dense
        assert isinstance(returned_architecture[0], ConvLayer)
        assert returned_architecture[0].filters == filters_first_dim
        assert returned_architecture[0].kernel_size == kernel_size_first_dim
        assert isinstance(returned_architecture[1], AvgPooling)
        assert isinstance(returned_architecture[2], ConvLayer)
        assert returned_architecture[2].filters == filters_second_dim
        assert returned_architecture[2].kernel_size == kernel_size_second_dim
        assert isinstance(returned_architecture[3], MaxPooling)
        assert isinstance(returned_architecture[4], ConvLayer)
        assert returned_architecture[4].filters == filters_third_dim
        assert returned_architecture[4].kernel_size == kernel_size_third_dim
        assert isinstance(returned_architecture[5], FlattenLayer)
        assert isinstance(returned_architecture[6], DenseLayer)
