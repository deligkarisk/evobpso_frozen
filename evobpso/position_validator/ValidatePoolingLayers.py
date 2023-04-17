import random

from evobpso.position_validator.PositionValidator import PositionValidator
from evobpso.utils.utils import extract_integer_from_subset_of_bits, clear_bit, set_bit


class ValidatePoolingLayers(PositionValidator):

    def __init__(self, pooling_layer_bit_num):
        self.pooling_layer_bit_num = pooling_layer_bit_num

    def validate(self, position, params):
        pooling_layers = [extract_integer_from_subset_of_bits(encoded_layer, start_index=self.pooling_layer_bit_num, length=1) for encoded_layer in position]
        pooling_layers_indexes = [idx for idx, value in enumerate(pooling_layers) if value == 1]
        num_pooling_layers = len(pooling_layers_indexes)
        if num_pooling_layers > params.architecture_params.max_pooling_layers:
            # first clear all pooling layer bits in the position
            for i in range(0, len(position)):
                position[i] = clear_bit(position[i], self.pooling_layer_bit_num)

            # then, randomly set the maximum number of bits (from the set of bits that were set before)
            for i in range(0, params.architecture_params.max_pooling_layers):
                layer_num = random.randint(0, len(pooling_layers_indexes) - 1)
                layer_val = pooling_layers_indexes[layer_num]
                position[layer_val] = set_bit(position[layer_val], self.pooling_layer_bit_num)
                pooling_layers_indexes.pop(layer_num)
        return position