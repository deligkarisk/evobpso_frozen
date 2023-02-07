from component_data_calculator.ComponentDataCalculator import ComponentDataCalculator
from utils.utils import create_rnd_binary_vector


class StandardBooleanComponentDataCalculator(ComponentDataCalculator):
    def calculate(self, best_position_data, current_position_data, c_factor):
        rnd_vector = create_rnd_binary_vector(c_factor, self.params.pso_params.n_bits)
        result = (best_position_data ^ current_position_data) & rnd_vector
        return result