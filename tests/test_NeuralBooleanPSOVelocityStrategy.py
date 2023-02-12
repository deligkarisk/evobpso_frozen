import copy
from unittest import TestCase
from unittest.mock import patch, Mock

from component_creator.StandardBooleanComponentCreator import StandardBooleanComponentCreator
from component_data_calculator.StandardBoolenComponentDataCalculator import StandardBooleanComponentDataCalculator
from params.Params import Params
from params.PsoParams import PsoParams, BooleanPSOParams
from velocity_factor.VelocityFactor import VelocityFactorRemove, VelocityFactorAdd, VelocityFactorEvolve
from velocity_update_strategy.StandardVelocityUpdateStrategy import StandardVelocityUpdateStrategy


class TestNeuralBooleanPSOStandardVelocityStrategy(TestCase):

    def test_get_new_velocity_only_personal_factor(self):
        current_velocity = [0b111111, 0b101010]
        current_position = [0b010101, 0b000000]
        pbest_position = [0b000000, 0b000000, 0b000111]
        gbest_position = [0b111111, 0b111111]

        pso_params = BooleanPSOParams(1, 0, 6, k=1)
        architecture_params = Mock()
        params = Params(pso_params, architecture_params)
        data_calculator = StandardBooleanComponentDataCalculator(params=params)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)

        strategy = StandardVelocityUpdateStrategy(component_creator=component_creator, params=params)
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
        pso_params = BooleanPSOParams(0, 1, 6, k=0)
        architecture_params = Mock()
        params = Params(pso_params, architecture_params)

        data_calculator = StandardBooleanComponentDataCalculator(params=params)
        component_creator = StandardBooleanComponentCreator(data_calculator=data_calculator)

        strategy = StandardVelocityUpdateStrategy(component_creator=component_creator, params=params)
        new_velocity = strategy.get_new_velocity(current_velocity, current_position, pbest_position, gbest_position)
        expected_velocity = []
        expected_velocity.append(VelocityFactorEvolve(data=0b101000))
        expected_velocity.append(VelocityFactorRemove())
        expected_velocity.append(VelocityFactorRemove())
        assert new_velocity == expected_velocity


    def test__equalize_sizes(self):
        mock_params = Mock()
        component_creator = Mock()
        velocity_strategy = StandardVelocityUpdateStrategy(component_creator, mock_params)

        personal_component = [VelocityFactorEvolve(data=0b000000),
                              VelocityFactorAdd(data=0b000000)]
        global_component = [VelocityFactorEvolve(data=0b000000),
                            VelocityFactorEvolve(data=0b000000),
                            VelocityFactorAdd(data=0b000000),
                            VelocityFactorAdd(data=0b000000)
                            ]

        expected_personal_component = copy.deepcopy(personal_component)
        expected_personal_component.append(VelocityFactorRemove())
        expected_personal_component.append(VelocityFactorRemove())
        expected_global_component = copy.deepcopy(global_component)

        updated_personal_component, updated_global_component = velocity_strategy._equalize_sizes(personal_component, global_component)

        assert expected_global_component == updated_global_component
        assert expected_personal_component == updated_personal_component
