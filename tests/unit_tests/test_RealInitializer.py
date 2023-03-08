from unittest import TestCase
from unittest.mock import patch, Mock

from initializer.RealInitializer import RealInitializer
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import RealPSOParams


class TestRealInitializer(TestCase):

    @patch('random.randint', return_value=100)
    def test_get_initial_position(self, mock_randint):
        pso_params = RealPSOParams(0, 0, 0, 1, 3, 0)
        architecture = NeuralArchitectureParams(2, 4)
        training_params = Mock()
        params = OptimizationParams(pso_params=pso_params, architecture_params=architecture, training_params=training_params)
        initializer = RealInitializer(params)
        new_position = initializer.get_initial_position()
        assert len(new_position) == 100
        smaller_than_min = [x for x in new_position if x < 1]
        larger_than_max = [x for x in new_position if x > 3]
        assert len(smaller_than_min) == 0
        assert len(larger_than_max) == 0


