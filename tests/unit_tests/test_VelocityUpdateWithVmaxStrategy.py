from unittest import TestCase
from unittest.mock import Mock, patch

from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorAdd
from evobpso.velocity_update_strategy.VelocityUpdateWithVmaxStrategy import VelocityUpdateWithVmaxStrategy


class TestVelocityUpdateWithVmaxStrategy(TestCase):

    @patch('evobpso.velocity_update_strategy.VelocityUpdateWithVmaxStrategy.super')
    def test_get_new_velocity_only_personal_factor(self, mock_super):

        # exact values of the position are irrelevant as we specify the return value of the super's get_new_velocity() a few lines below.
        current_position = Mock()
        pbest_position = Mock()
        gbest_position = Mock()

        velocity = []
        velocity.append(VelocityFactorEvolve(data=0b010101))
        velocity.append(VelocityFactorEvolve(data=0b000000))
        velocity.append(VelocityFactorAdd(data=0b000111))
        mock_super().get_new_velocity.return_value = velocity

        params = Mock()
        strategy = VelocityUpdateWithVmaxStrategy(component_creator=Mock(), component_merger=Mock(), params=params)

        # as the vmax process depends on random numbers, we should do the assertions many times
        # try first with vmax = 2
        params.pso_params.vmax = 2
        for i in range(0, 100):
            new_velocity = strategy.get_new_velocity(current_position, pbest_position, gbest_position)
            vmax_data_layer_one = new_velocity[0].data
            vmax_data_layer_two = new_velocity[1].data
            vmax_data_layer_three = new_velocity[2].data

            # Check the first two velocity layers for correct number of 1 bits The third layer is an Add layer, so
            # the same constraint does not apply
            assert vmax_data_layer_one.bit_count() <= params.pso_params.vmax
            assert vmax_data_layer_two.bit_count() <= params.pso_params.vmax

        # try again with vmax = 3
        params.pso_params.vmax = 3

        # as the vmax process depends on random numbers, we should do the assertions many times
        for i in range(0, 100):
            new_velocity = strategy.get_new_velocity(current_position, pbest_position, gbest_position)
            vmax_data_layer_one = new_velocity[0].data
            vmax_data_layer_two = new_velocity[1].data
            vmax_data_layer_three = new_velocity[2].data

            # no new velocity should have more than vmax one digits
            # Check the first two velocity layers for correct number of 1 bits The third layer is an Add layer, so
            # the same constraint does not apply
            assert vmax_data_layer_one.bit_count() <= params.pso_params.vmax
            assert vmax_data_layer_two.bit_count() <= params.pso_params.vmax
