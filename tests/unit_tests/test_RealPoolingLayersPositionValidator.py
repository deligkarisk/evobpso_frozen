import time
from unittest import TestCase
from unittest.mock import Mock

from evobpso.position_validator.BooleanPoolingLayersPositionValidator import BooleanPoolingLayersPositionValidator
from evobpso.position_validator.RealPoolingLayersPositionValidator import RealPoolingLayersPositionValidator
from evobpso.utils.utils import extract_integer_from_subset_of_bits


class TestRealPoolingLayersPositionValidator(TestCase):
    def test_validate(self):
        position = []
        position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.8})
        position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.88})
        position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.34})
        position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.65})
        position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.2})
        position.append({'conv_filters': 131, 'kernel_size': 5, 'pooling': 0.})

        params = Mock()
        params.neural_architecture_params.max_pooling_layers = 2
        validator = RealPoolingLayersPositionValidator()

        pooling_layers = [encoded_layer['pooling'] for
                          encoded_layer in position]
        pooling_layers_indexes = [idx for idx, value in enumerate(pooling_layers) if value > 0.33]
        num_pooling_layers = len(pooling_layers_indexes)

        assert num_pooling_layers == 4  # As initialized, four layers have a pooling layer, i.e. the 11th bit (starting from zero) is 1
        new_position = validator.validate(position, params)

        new_pooling_layers = [encoded_layer['pooling'] for
                          encoded_layer in new_position]
        new_pooling_layers_indexes = [idx for idx, value in enumerate(new_pooling_layers) if value > 0.33]
        new_num_pooling_layers = len(new_pooling_layers_indexes)

        assert new_num_pooling_layers == params.neural_architecture_params.max_pooling_layers

        # new pooling layers indexes should be a subset of the original ones
        assert set(new_pooling_layers_indexes) <= set(pooling_layers_indexes)
