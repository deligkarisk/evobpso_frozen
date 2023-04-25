import copy
from unittest import TestCase
from unittest.mock import patch, Mock

from evobpso.component_creator.VariableLengthComponentCreator import VariableLengthComponentCreator
from evobpso.component_creator_data_calculator.BoolenComponentCreatorDataCalculator import BooleanComponentCreatorDataCalculator
from evobpso.params.OptimizationParams import OptimizationParams
from evobpso.params.PsoParams import PsoParams, BooleanPSOParams
from evobpso.velocity_factor.VelocityFactor import VelocityFactorRemove, VelocityFactorAdd, VelocityFactorEvolve
from evobpso.velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy
from evobpso.velocity_update_strategy.component_merge_strategy.VariableLengthCalculateDataComponentMergeStrategy import VariableLengthCalculateDataComponentMergeStrategy
from evobpso.velocity_update_strategy.component_merge_strategy.component_merger_data_calculator.BooleanComponentMergerDataCalculator import \
    BooleanComponentMergerDataCalculator


class VariableLengthBooleanPSOVelocityUpdateCase(TestCase):

    def test_get_new_velocity_only_personal_factor(self):

        # This test simulates the velocity update strategy with variable length position, and boolean operations

        current_velocity = [0b111111, 0b101010]
        current_position = [0b010101, 0b000000]
        pbest_position = [0b000000, 0b000000, 0b000111]
        gbest_position = [0b111111, 0b111111]

        pso_params = BooleanPSOParams(pop_size=2, iters=1, c1=1, c2=0, n_bits=6, k=1, mutation_prob=0)
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

    def test_get_new_velocity_only_global_factor(self):
        current_velocity = [0b111111, 0b101010]
        current_position = [0b010101, 0b000000]
        pbest_position = [0b000000, 0b000000, 0b000111]
        gbest_position = [0b111101]
        pso_params = BooleanPSOParams(pop_size=10, iters=10,
                                      n_bits=6,
                                      c1=0, c2=1, k=0, mutation_prob=0)
        architecture_params = Mock()
        training_params = Mock()
        params = OptimizationParams(pso_params, architecture_params, training_params)

        data_calculator = BooleanComponentCreatorDataCalculator(params=params)
        component_creator = VariableLengthComponentCreator(data_calculator=data_calculator)
        component_merger_data_calculator = BooleanComponentMergerDataCalculator()

        component_merger = VariableLengthCalculateDataComponentMergeStrategy(component_merger_data_calculator=component_merger_data_calculator)

        strategy = StandardVelocityUpdateStrategy(component_creator=component_creator,
                                                  component_merger=component_merger,
                                                  params=params)
        new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position)
        expected_velocity = []
        expected_velocity.append(VelocityFactorEvolve(data=0b101000))
        expected_velocity.append(VelocityFactorRemove())
        expected_velocity.append(VelocityFactorRemove())
        assert new_velocity == expected_velocity



