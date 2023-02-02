from unittest import TestCase
from unittest.mock import patch

from initializer.BinaryInitializer import BinaryInitializer
from problem.NeuralArchitecture import NeuralArchitecture
from pso_params.PsoParams import BooleanPSOParams


class TestBinaryInitializer(TestCase):

    @patch('random.randint', return_value=4)
    @patch('initializer.BinaryInitializer.create_rnd_binary_vector')
    def test_get_initial_position(self, mock_rnd_vector_function, mock_rand):
        params = BooleanPSOParams(0, 0, 0, 8, 0)
        architecture = NeuralArchitecture(2, 4, 2, 4, 2, 10)
        initializer = BinaryInitializer(architecture, params)
        new_position = initializer.get_initial_position()
        assert mock_rnd_vector_function.assert_called
        assert mock_rnd_vector_function.call_count == 4
