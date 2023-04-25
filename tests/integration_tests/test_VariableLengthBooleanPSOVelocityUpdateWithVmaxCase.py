import copy
from unittest import TestCase
from unittest.mock import patch, Mock

from evobpso.component_creator.VariableLengthComponentCreator import VariableLengthComponentCreator
from evobpso.component_creator_data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import PsoParams, BooleanPSOParams
from evobpso.velocity_factor.VelocityFactor import VelocityFactorRemove, VelocityFactorAdd, VelocityFactorEvolve
from evobpso.velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy
from evobpso.velocity_update_strategy.VelocityUpdateWithVmaxStrategy import VelocityUpdateWithVmaxStrategy
from evobpso.velocity_update_strategy.component_merge_strategy.VariableLengthCalculateDataComponentMergeStrategy import VariableLengthCalculateDataComponentMergeStrategy
from evobpso.velocity_update_strategy.component_merge_strategy.component_merger_data_calculator.BooleanComponentMergerDataCalculator import \
    BooleanComponentMergerDataCalculator


class VariableLengthBooleanPSOVelocityUpdateWithVmaxCase(TestCase):

    def test_get_new_velocity_only_personal_factor(self):
        current_velocity = [0b111111, 0b101010]
        current_position = [0b010101, 0b000000]
        pbest_position = [0b000000, 0b000000, 0b000111]
        gbest_position = [0b111111, 0b111111]


        # first try with vmax = 0 to get the results as they should be without vmax
        pso_params = BooleanPSOParams(2, 1, 1, 0, 6, k=1, mutation_prob=0, vmax=0 )
        architecture_params = Mock()
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)
        data_calculator = BooleanComponentCreatorDataCalculator(params=params)
        component_creator = VariableLengthComponentCreator(data_calculator=data_calculator)
        component_merger_data_calculator = BooleanComponentMergerDataCalculator()

        component_merger = VariableLengthCalculateDataComponentMergeStrategy(component_merger_data_calculator=component_merger_data_calculator)

        strategy = StandardVelocityUpdateStrategy(component_creator=component_creator, component_merger=component_merger, params=params)
        new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position)
        expected_velocity = []
        expected_velocity.append(VelocityFactorEvolve(data=0b010101))
        expected_velocity.append(VelocityFactorEvolve(data=0b000000))
        expected_velocity.append(VelocityFactorAdd(data=0b000111))
        assert new_velocity == expected_velocity

        # get a copy of the data before running with the vmax set
        standard_data_layer_one = expected_velocity[0].data
        standard_data_layer_two = expected_velocity[1].data
        standard_data_layer_three = expected_velocity[2].data

        # now re-run the same configuration, but with vmax set, to count the number of ones
        strategy = VelocityUpdateWithVmaxStrategy(component_creator=component_creator, component_merger=component_merger, params=params)
        pso_params.vmax = 2

        # as the vmax process depends on random numbers, we should do the assertions many times
        for i in range(0, 100):
            new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position)
            vmax_data_layer_one = new_velocity[0].data
            vmax_data_layer_two = new_velocity[1].data
            vmax_data_layer_three = new_velocity[2].data


            # no new velocity should have more than vmax one digits
            # Check the first two velocity layers for correct number of 1 bits The third layer is an Add layer, so
            # the same constraint does not apply
            assert vmax_data_layer_one.bit_count() <= pso_params.vmax
            assert vmax_data_layer_two.bit_count() <= pso_params.vmax

            assert vmax_data_layer_two == standard_data_layer_two
