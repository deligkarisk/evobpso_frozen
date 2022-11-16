from unittest import TestCase
from unittest.mock import patch

from velocity_strategy.BooleanPSOVelocityStrategy import BooleanPSOStandardVelocityStrategy

class TestBooleanPSOStandardVelocityStrategy(TestCase):

    @patch('pso_params.PsoParams.StandardPsoParams')
    def test_update_velocity_local_factor_only(self, mock_params):
        current_velocity = [0b000000]
        current_position = [0b000000]
        personal_best_position = [0b111111]
        global_best_position = [0b010101]

        mock_params.c1 = 1
        mock_params.c2 = 0
        mock_params.omega = 0
        mock_params.n_bits = 6

        velocity_strategy = BooleanPSOStandardVelocityStrategy()
        returned_velocity = velocity_strategy.get_new_velocity(current_velocity, current_position, personal_best_position, global_best_position, mock_params)
        assert returned_velocity == [0b111111]


    @patch('pso_params.PsoParams.StandardPsoParams')
    def test_update_velocity_global_factor_only(self, mock_params):
        current_velocity = [0b000000]
        current_position = [0b000000]
        personal_best_position = [0b111111]
        global_best_position = [0b010101]

        mock_params.c1 = 0
        mock_params.c2 = 1
        mock_params.omega = 0
        mock_params.n_bits = 6

        velocity_strategy = BooleanPSOStandardVelocityStrategy()
        returned_velocity = velocity_strategy.get_new_velocity(current_velocity, current_position, personal_best_position, global_best_position, mock_params)
        assert returned_velocity == [0b010101]
