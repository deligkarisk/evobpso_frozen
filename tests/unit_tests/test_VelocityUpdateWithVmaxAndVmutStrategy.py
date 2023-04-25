from unittest import TestCase
from unittest.mock import patch, Mock

from evobpso.velocity_factor.VelocityFactor import VelocityFactorAdd, VelocityFactorEvolve
from evobpso.velocity_update_strategy.VelocityUpdateWithVmaxAndVmutStrategy import VelocityUpdateWithVmaxAndVmutStrategy


class TestVelocityUpdateWithVmaxAndVmutStrategy(TestCase):

    @patch('evobpso.velocity_update_strategy.VelocityUpdateWithVmaxAndVmutStrategy.super')
    @patch('evobpso.velocity_update_strategy.VelocityUpdateWithVmaxAndVmutStrategy.create_rnd_binary_vector', return_value=0b100000)
    def test_get_new_velocity(self, mock_rnd_binary_vector_creator, mock_super):
        # exact values of the position are irrelevant as we specify the return value of the super's get_new_velocity() a few lines below.
        current_position = Mock()
        pbest_position = Mock()
        gbest_position = Mock()

        base_velocity = []
        base_velocity.append(VelocityFactorEvolve(data=0b010101))
        base_velocity.append(VelocityFactorEvolve(data=0b000000))
        base_velocity.append(VelocityFactorAdd(data=0b000111))
        mock_super().get_new_velocity.return_value = base_velocity

        strategy = VelocityUpdateWithVmaxAndVmutStrategy(component_creator=Mock(), component_merger=Mock(),
                                                         params=Mock())
        new_velocity = strategy.get_new_velocity(current_position, pbest_position, gbest_position)
        expected_velocity = []
        expected_velocity.append(VelocityFactorEvolve(data=0b110101))  # same as the base velocity , but with the left-most bit now set
        # to 1 (due to the create_rnd_binary_vector return value being set)
        expected_velocity.append(VelocityFactorEvolve(data=0b100000))  # same as the base velocity, but with the left-most bit now set to
        # 1 (due to the create_rnd_binary_vector return value being set)
        expected_velocity.append(VelocityFactorAdd(data=0b100111))  # same as the base velocity, but with the left-most bit now set to 1
        # (due to the create_rnd_binary_vector return value being set)
        assert new_velocity == expected_velocity


