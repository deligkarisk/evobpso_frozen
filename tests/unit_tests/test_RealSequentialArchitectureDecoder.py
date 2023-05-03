from unittest import TestCase

from evobpso.architecture_decoder.BooleanSequentialArchitectureDecoder import BooleanSequentialArchitectureDecoder
from evobpso.architecture_decoder.RealSequentialArchitectureDecoder import RealSequentialArchitectureDecoder
from evobpso.encoding.BooleanEncoding import BooleanEncoding
from evobpso.encoding.RealEncoding import RealEncoding
from evobpso.layer.Layer import AvgPooling, MaxPooling, ConvLayer, FlattenLayer, DenseLayer


class TestRealSequentialArchitectureDecoder(TestCase):
    def test_decode_with_boolean_sample_encoding(self):

        encoding = RealEncoding(min_conv_filters=2, max_conv_filters=12, min_kernel_size=2, max_kernel_size=3)
        position = []
        position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.88})
        filters_first_dim = 131  # binary is 130, then adding one as the minimum filter is one, not zero.
        kernel_size_first_dim = 5  # binary is three, adding two as the minimum kernel size is two.
        pooling_layer_type_first_dim = AvgPooling  # a value of one is average pooling
        position.append({'conv_filters': 19, 'kernel_size': 9, 'pooling': 0.6})
        filters_second_dim = 19  # binary is 18, then adding one as the minimum filter is one, not zero.
        kernel_size_second_dim = 9  # binary is three, adding two as the minimum kernel size is two.
        pooling_layer_type_second_dim = MaxPooling  # a value of zero is max pooling
        position.append({'conv_filters': 4, 'kernel_size': 2, 'pooling': 0.33})
        filters_third_dim = 4  # binary is three, then adding one as the minimum filter is one, not zero.
        kernel_size_third_dim = 2  # binary is zero, adding two as the minimum kernel size is two.

        decoder = RealSequentialArchitectureDecoder(encoding=encoding)
        returned_architecture = decoder.decode(position)

        assert len(returned_architecture) == 7  # Conv, Pooling, Conv, Pooling, Conv, Flatten, Dense
        assert isinstance(returned_architecture[0], ConvLayer)
        assert returned_architecture[0].filters == filters_first_dim
        assert returned_architecture[0].kernel_size == kernel_size_first_dim
        assert isinstance(returned_architecture[1], pooling_layer_type_first_dim)
        assert isinstance(returned_architecture[2], ConvLayer)
        assert returned_architecture[2].filters == filters_second_dim
        assert returned_architecture[2].kernel_size == kernel_size_second_dim
        assert isinstance(returned_architecture[3], pooling_layer_type_second_dim)
        assert isinstance(returned_architecture[4], ConvLayer)
        assert returned_architecture[4].filters == filters_third_dim
        assert returned_architecture[4].kernel_size == kernel_size_third_dim
        assert isinstance(returned_architecture[5], FlattenLayer)
        assert isinstance(returned_architecture[6], DenseLayer)
