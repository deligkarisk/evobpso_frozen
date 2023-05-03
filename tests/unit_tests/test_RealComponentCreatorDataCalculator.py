from unittest import TestCase
from unittest.mock import Mock, patch

from evobpso.component_creator.data_calculator.RealComponentCreatorDataCalculator import RealComponentCreatorDataCalculator
from evobpso.params.OptimizationParams import OptimizationParams


class TestRealComponentCreatorDataCalculator(TestCase):

    @patch('evobpso.component_creator.data_calculator.RealComponentCreatorDataCalculator.random.random', return_value=1)
    def test_calculate_with_c_equal_to_one(self, mock_random):
        pso_params = Mock()
        architecture_params = Mock()
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)

        best_position_data = {'conv_filters': 12, 'kernel_size': 3, 'pooling': 0.3}
        current_position_data = {'conv_filters': 4, 'kernel_size': 5, 'pooling': 0.9}

        calculator = RealComponentCreatorDataCalculator(params=params)
        result = calculator.calculate(best_position_data, current_position_data, c_factor=1)
        assert result['conv_filters'] == 8
        assert result['kernel_size'] == -2
        assert -0.60001 < result['pooling'] < -0.5999

    @patch('evobpso.component_creator.data_calculator.RealComponentCreatorDataCalculator.random.random', return_value=1)
    def test_calculate_with_c_equal_to_zero(self, mock_random):
        pso_params = Mock()
        architecture_params = Mock()
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)

        best_position_data = {'conv_filters': 12, 'kernel_size': 3, 'pooling': 0.3}
        current_position_data = {'conv_filters': 4, 'kernel_size': 5, 'pooling': 0.9}

        calculator = RealComponentCreatorDataCalculator(params=params)
        result = calculator.calculate(best_position_data, current_position_data, c_factor=0)
        assert result['conv_filters'] == 0
        assert result['kernel_size'] == 0
        assert result['pooling'] == 0



