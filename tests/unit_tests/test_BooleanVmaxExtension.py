from unittest import TestCase
from unittest.mock import Mock, patch

from evobpso.velocity_factor.VelocityFactor import VelocityFactorEvolve, VelocityFactorAdd
from evobpso.velocity_update_extension.BooleanVmaxExtension import BooleanVmaxExtension


class TestBooleanVmaxExtension(TestCase):

    def test_get_new_velocity(self):

        base_velocity = []
        base_velocity.append(VelocityFactorEvolve(data=0b010101))
        base_velocity.append(VelocityFactorEvolve(data=0b000000))
        base_velocity.append(VelocityFactorAdd(data=0b000111))

        params = Mock()
        extension = BooleanVmaxExtension(params)

        # as the vmax process depends on random numbers, we should do the assertions many times
        # try first with vmax = 2
        params.pso_params.vmax = 2
        for i in range(0, 100):
            new_velocity = extension.get_new_velocity(base_velocity)
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
            new_velocity = extension.get_new_velocity(base_velocity)
            vmax_data_layer_one = new_velocity[0].data
            vmax_data_layer_two = new_velocity[1].data
            vmax_data_layer_three = new_velocity[2].data

            # no new velocity should have more than vmax one digits
            # Check the first two velocity layers for correct number of 1 bits The third layer is an Add layer, so
            # the same constraint does not apply
            assert vmax_data_layer_one.bit_count() <= params.pso_params.vmax
            assert vmax_data_layer_two.bit_count() <= params.pso_params.vmax
