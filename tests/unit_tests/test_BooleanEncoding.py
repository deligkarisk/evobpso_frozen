from unittest import TestCase

from evobpso.encoding.BooleanEncoding import BooleanEncoding


class TestBooleanEncoding(TestCase):

    def test_construction(self):
        encoding = BooleanEncoding(filter_bits=8, kernel_size_bits=3)

        # the standard architecture is with one bit for pooling layer and type
        assert encoding.pooling_type_bits == 1
        assert encoding.pooling_layer_bits == 1

        assert encoding.pooling_layer_bit_position == 11
        assert encoding.pooling_type_bit_position == 12

        assert encoding.total_bits == 13  # 11 for filter bits and kernel size plus two for pooling layer and type


        encoding = BooleanEncoding(filter_bits=4, kernel_size_bits=2)

        # the standard architecture is with one bit for pooling layer and type
        assert encoding.pooling_type_bits == 1
        assert encoding.pooling_layer_bits == 1

        assert encoding.pooling_layer_bit_position == 6
        assert encoding.pooling_type_bit_position == 7

        assert encoding.total_bits == 8  # 6 for filter bits and kernel size plus two for pooling layer and type
