from unittest import TestCase
from unittest.mock import Mock

from evobpso.position_validator.ValidatePoolingLayers import ValidatePoolingLayers
from evobpso.utils.utils import extract_integer_from_subset_of_bits


class TestValidatePoolingLayers(TestCase):
    def test_validate(self):
        pooling_layer_bit_num = 11
        position = [0b0100000000001, 0b0100000000010, 0b0100000000100, 0b0100000001000, 0b0000000010000, 0b0000000010000]
        params = Mock()
        params.architecture_params.max_pooling_layers = 2
        validator = ValidatePoolingLayers(pooling_layer_bit_num=pooling_layer_bit_num)

        pooling_layers = [extract_integer_from_subset_of_bits(encoded_layer, start_index=pooling_layer_bit_num, length=1) for
                          encoded_layer in position]
        pooling_layers_indexes = [idx for idx, value in enumerate(pooling_layers) if value == 1]
        num_pooling_layers = len(pooling_layers_indexes)

        assert num_pooling_layers == 4  # As initialized, four layers have a pooling layer, i.e. the 11th bit (starting from zero) is 1

        new_position = validator.validate(position, params)
        new_pooling_layers = [extract_integer_from_subset_of_bits(new_encoded_layer, start_index=pooling_layer_bit_num, length=1) for
                              new_encoded_layer in new_position]
        new_pooling_layers_indexes = [idx for idx, value in enumerate(new_pooling_layers) if value == 1]
        new_num_pooling_layers = len(new_pooling_layers_indexes)

        assert new_num_pooling_layers == params.architecture_params.max_pooling_layers

        # new pooling layers indexes should be a subset of the original ones
        assert set(new_pooling_layers_indexes) <= set(pooling_layers_indexes)
