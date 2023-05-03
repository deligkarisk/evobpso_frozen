import random

from evobpso.component_creator.data_calculator.ComponentCreatorDataCalculator import ComponentCreatorDataCalculator


class RealComponentCreatorDataCalculator(ComponentCreatorDataCalculator):
    def calculate(self, best_position_data, current_position_data, c_factor):
        result = {}
        rnd_value = random.random()
        result['conv_filters'] = int((best_position_data['conv_filters'] - current_position_data['conv_filters']) * rnd_value * c_factor)
        rnd_value = random.random()
        result['kernel_size'] = int((best_position_data['kernel_size'] - current_position_data['kernel_size']) * rnd_value * c_factor)
        rnd_value = random.random()
        result['pooling'] = (best_position_data['pooling'] - current_position_data['pooling']) * rnd_value * c_factor
        return result
