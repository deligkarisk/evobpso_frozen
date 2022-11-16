from unittest import TestCase
import random
from unittest.mock import patch

from velocity_strategy.RealPSOVelocityStrategy import RealPSOStandardVelocityStrategy


class TestRealPSOStandardVelocityStrategy(TestCase):

    @patch('pso_params.PsoParams.StandardPsoParams')
    @patch('random.uniform')
    def test_get_new_velocity(self, mock_random, mock_params):
        mock_random.return_value = 1


        current_velocity = [0.1]
        current_position = [2]
        personal_best_position = [4]
        global_best_position = [1]
        mock_params.c1 = 1
        mock_params.c2 = 0
        mock_params.omega = 0


        velocity_strategy = RealPSOStandardVelocityStrategy()
        returned_velocity = velocity_strategy.get_new_velocity(current_velocity, current_position, personal_best_position, global_best_position, mock_params)
        assert returned_velocity == [2.0]



