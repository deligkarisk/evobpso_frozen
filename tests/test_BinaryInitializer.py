from unittest import TestCase
from unittest.mock import patch

from initializer.BinaryInitializer import BinaryInitializer
from params.NeuralArchitectureParams import NeuralArchitectureParams
from params.OptimizationParams import OptimizationParams
from params.PsoParams import BooleanPSOParams


class TestBinaryInitializer(TestCase):

    @patch('random.randint', return_value=4)
    @patch('initializer.BinaryInitializer.create_rnd_binary_vector')
    def test_get_initial_position(self, mock_rnd_vector_function, mock_rand):
        pso_params = BooleanPSOParams(0, 0, 8, 0)
        architecture = NeuralArchitectureParams(2, 10)
        params = OptimizationParams(pso_params=pso_params, architecture_params=architecture)
        initializer = BinaryInitializer(params)
        new_position = initializer.get_initial_position()
        assert mock_rnd_vector_function.assert_called
        assert mock_rnd_vector_function.call_count == 4
