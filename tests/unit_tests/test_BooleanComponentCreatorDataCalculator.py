from unittest import TestCase
from unittest.mock import Mock

from evobpso.component_creator_data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.params.OptimizationParams import OptimizationParams


class TestBooleanComponentCreatorDataCalculator(TestCase):

    def test_calculate_with_c_equal_to_one(self):
        pso_params = Mock()
        architecture_params = Mock()
        pso_params.n_bits = 8
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)

        best_position_data = 0b01010111
        current_position_data = 0b11111111

        calculator = BooleanComponentCreatorDataCalculator(params=params)
        result = calculator.calculate(best_position_data, current_position_data, c_factor=1)
        assert result == 0b10101000

    def test_calculate_with_c_equal_to_zero(self):
        pso_params = Mock()
        architecture_params = Mock()
        pso_params.n_bits = 8
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)

        best_position_data = 0b01010111
        current_position_data = 0b11111111

        calculator = BooleanComponentCreatorDataCalculator(params=params)
        result = calculator.calculate(best_position_data, current_position_data, c_factor=0)
        assert result == 0b00000000



