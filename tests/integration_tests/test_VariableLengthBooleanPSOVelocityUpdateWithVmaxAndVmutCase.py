import time
from unittest import TestCase
from unittest.mock import patch, Mock

from evobpso.component_creator.VariableLengthComponentCreator import VariableLengthComponentCreator
from evobpso.component_creator.data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import BooleanPSOParams
from evobpso.velocity_factor.VelocityFactor import VelocityFactorAdd, VelocityFactorEvolve
from evobpso.velocity_update_strategy.VelocityUpdateWithVmaxAndVmutStrategy import VelocityUpdateWithVmaxAndVmutStrategy
from evobpso.velocity_update_strategy.VelocityUpdateWithVmaxStrategy import VelocityUpdateWithVmaxStrategy
from evobpso.component_merger.VariableLengthCalculateDataComponentMerger import VariableLengthCalculateDataComponentMerger
from evobpso.component_merger.data_calculator.BooleanComponentMergerDataCalculator import \
    BooleanComponentMergerDataCalculator


class VariableLengthBooleanPSOVelocityUpdateWithVmaxAndVmutCase(TestCase):

    @patch('evobpso.velocity_update_strategy.VelocityUpdateWithVmaxAndVmutStrategy.create_rnd_binary_vector', return_value=0b100000)
    def test_get_new_velocity_only_personal_factor(self, mock_rnd_binary_vector_creator):

        current_velocity = [0b111111, 0b101010]
        current_position = [0b010101, 0b000000]
        pbest_position = [0b000000, 0b000000, 0b000111]
        gbest_position = [0b111111, 0b111111]

        # first try with vmax = 0 to get the results as they should be without vmax
        pso_params = BooleanPSOParams(2, 1, 1, 0, 6, k=1, mutation_prob=0, vmax=0)
        architecture_params = Mock()
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)
        data_calculator = BooleanComponentCreatorDataCalculator(params=params)
        component_creator = VariableLengthComponentCreator(data_calculator=data_calculator)
        component_merger_data_calculator = BooleanComponentMergerDataCalculator()

        component_merger = VariableLengthCalculateDataComponentMerger(component_merger_data_calculator=component_merger_data_calculator)

        # now re-run the same configuration, but with vmax set, to count the number of ones

        strategy = VelocityUpdateWithVmaxStrategy(component_creator=component_creator, component_merger=component_merger, params=params)
        pso_params.vmax = 6 # set to 6, so we do not have any 1 turned to 0 (i.e. make the output deterministic)
        start_time = time.time()
        new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position)
        end_time = time.time()
        elapsed_time = ((end_time - start_time))
        print("Position validator runtime: " + str(elapsed_time) + " seconds.")
        expected_velocity = []
        expected_velocity.append(VelocityFactorEvolve(data=0b010101))
        expected_velocity.append(VelocityFactorEvolve(data=0b000000))
        expected_velocity.append(VelocityFactorAdd(data=0b000111))
        assert new_velocity == expected_velocity

        strategy = VelocityUpdateWithVmaxAndVmutStrategy(component_creator=component_creator, component_merger=component_merger, params=params)
        start_time = time.time()
        new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position)
        end_time = time.time()
        elapsed_time = ((end_time - start_time))
        print("Position validator runtime: " + str(elapsed_time) + " seconds.")
        expected_velocity = []
        expected_velocity.append(VelocityFactorEvolve(data=0b110101))  # same as before, but with the left-most bit now set to 1 (due to
        # the create_rnd_binary_vector return value being set
        expected_velocity.append(VelocityFactorEvolve(data=0b100000))   # same as before, but with the left-most bit now set to 1 (due to
        # the create_rnd_binary_vector return value being set
        expected_velocity.append(VelocityFactorAdd(data=0b100111))   # same as before, but with the left-most bit now set to 1 (due to
        # the create_rnd_binary_vector return value being set
        assert new_velocity == expected_velocity
