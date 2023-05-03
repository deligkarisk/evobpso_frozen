from evobpso.position_validator.PositionValidator import PositionValidator
import random


class RealPoolingLayersPositionValidator(PositionValidator):


    def validate(self, position, params):
        pooling_layers = [encoded_layer['pooling'] for encoded_layer in position]
        pooling_layers_indexes = [idx for idx, value in enumerate(pooling_layers) if value > 0.33]
        num_pooling_layers = len(pooling_layers_indexes)
        if num_pooling_layers > params.neural_architecture_params.max_pooling_layers:
            # first clear all pooling layer bits in the position
            for i in range(0, len(position)):
                position[i]['pooling'] = 0

            # then, randomly set the maximum number of bits (from the set of bits that were set before)
            for i in range(0, params.neural_architecture_params.max_pooling_layers):
                layer_num = random.randint(0, len(pooling_layers_indexes) - 1)
                layer_val = pooling_layers_indexes[layer_num]
                position[layer_val]['pooling'] = random.uniform(0.33, 1)
                pooling_layers_indexes.pop(layer_num)
        return position