from unittest import TestCase
from unittest.mock import patch, Mock

from evobpso.initializer.BinaryInitializer import BinaryInitializer
from evobpso.params.NeuralArchitectureParams import NeuralArchitectureParams
from evobpso.params.OptimizationParams import OptimizationParams


class TestBinaryInitializer(TestCase):

    @patch('random.randint', return_value=4)
    @patch('evobpso.initializer.BinaryInitializer.create_rnd_binary_vector')
    def test_get_initial_position(self, mock_rnd_vector_function, mock_rand):
        pso_params = Mock()
        training_params = Mock()
        architecture = NeuralArchitectureParams(2, 10, max_pooling_layers=2)
        params = OptimizationParams(pso_params=pso_params, neural_architecture_params=architecture, training_params=training_params)
        encoding = Mock()
        initializer = BinaryInitializer(params, encoding=encoding)
        new_position = initializer.get_initial_position()
        assert mock_rnd_vector_function.assert_called
        assert mock_rnd_vector_function.call_count == 4
